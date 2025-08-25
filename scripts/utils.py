import os
import random
import numpy as np
from functools import partial

import timm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import torchmetrics
from transformers import AutoModel, AutoTokenizer

from scripts.dataset import MultimodalDataset, collate_fn, get_transforms


def seed_everything(seed: int):
    """
    Устанавливает seed для всех генераторов случайных чисел
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def set_requires_grad(module: nn.Module, unfreeze_pattern="", verbose=False):
    """
    Устанавливает requires_grad для параметров модели в зависимости от кон
    unfreeze_pattern: ключи, для которых requires_grad = True
    verbose: выводить ли информацию о размороженных слоях
    """
    if len(unfreeze_pattern) == 0:
        for _, param in module.named_parameters():
            param.requires_grad = False
        return

    pattern = unfreeze_pattern.split("|")

    for name, param in module.named_parameters():
        if any([name.startswith(p) for p in pattern]):
            param.requires_grad = True
            if verbose:
                print(f"Разморожен слой: {name}")
        else:
            param.requires_grad = False


class MultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Модели
        self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.image_model = timm.create_model(
            config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0
        )

        # Проекции
        self.text_proj = nn.Linear(self.text_model.config.hidden_size, config.HIDDEN_DIM)
        self.image_proj = nn.Linear(self.image_model.num_features, config.HIDDEN_DIM)

        self.regressor = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM * 2, config.HIDDEN_DIM // 2),
            nn.LayerNorm(config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM // 2, 1),
            nn.Softplus()
        )

    def forward(self, input_ids, attention_mask, image):
        # признаки текста
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = text_outputs.last_hidden_state
        text_mask = attention_mask.unsqueeze(-1).float()
        text_masked_embeddings = text_embeddings * text_mask
        text_features = text_masked_embeddings.sum(1) / text_mask.sum(1)

        # признаки изображения
        image_features = self.image_model(image)

        text_emb = self.text_proj(text_features)
        image_emb = self.image_proj(image_features)

        fused_emb = torch.cat([text_emb, image_emb], dim=1)

        logits = self.regressor(fused_emb)
        return logits


def train(config, device):
    seed_everything(config.SEED)

    # Инициализация модели
    model = MultimodalModel(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    set_requires_grad(model.text_model, unfreeze_pattern=config.TEXT_MODEL_UNFREEZE, verbose=True)
    set_requires_grad(model.image_model, unfreeze_pattern=config.IMAGE_MODEL_UNFREEZE, verbose=True)

    # Оптимизатор с разными LR
    optimizer = AdamW([{
        'params': model.text_model.parameters(),
        'lr': config.TEXT_LR
    }, {
        'params': model.image_model.parameters(),
        'lr': config.IMAGE_LR
    }, {
        'params': model.regressor.parameters(),
        'lr': config.REGRESSOR_LR
    }])

    criterion = nn.SmoothL1Loss(beta=20)

    # Загрузка данных
    transforms = get_transforms(config)
    val_transforms = get_transforms(config, ds_type="val")

    train_dataset = MultimodalDataset(config, transforms)
    val_dataset = MultimodalDataset(config, val_transforms, ds_type="val")

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=partial(collate_fn, tokenizer=tokenizer))
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer))

    # инициализируем метрику
    mae_metric_train = torchmetrics.MeanAbsoluteError().to(device)
    mae_metric_val = torchmetrics.MeanAbsoluteError().to(device)
    
    best_mae_val = float('inf')

    print("Training started")
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            # Подготовка данных
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device)
            }
            values = batch['value'].to(device)

            # Forward
            optimizer.zero_grad()
            logits = model(**inputs)
            loss = criterion(logits.squeeze(), values)

            # Backward
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            predicted = logits.squeeze()
            _ = mae_metric_train(preds=predicted, target=values)

        # Валидация
        train_mae = mae_metric_train.compute().cpu().numpy()
        val_mae = validate(model, val_loader, device, mae_metric_val)
        mae_metric_val.reset()
        mae_metric_train.reset()

        print(
            f"Epoch {epoch}/{config.EPOCHS-1} | Avg. Loss: {total_loss/len(train_loader):.4f} | Train MAE: {train_mae :.4f} | Val MAE: {val_mae :.4f}"
        )

        if val_mae < best_mae_val:
            print(f"New best model, epoch: {epoch}")
            best_mae_val = val_mae
            torch.save(model.state_dict(), config.SAVE_PATH)


def validate(model, val_loader, device, mae_metric):
    model.eval()

    with torch.no_grad():
        for batch in val_loader:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device)
            }
            values = batch['value'].to(device)

            logits = model(**inputs)
            predicted = logits.squeeze()
            _ = mae_metric(preds=predicted, target=values)

    return mae_metric.compute().cpu().numpy()


def load_model(config, model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model from {model_path}")
    model = MultimodalModel(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model


def inference_and_evaluate(model, dataset, config, device, batch_size=32):
    """
    Выполняет инференс на датасете и вычисляет метрики
    Возвращает предсказания, реальные значения, ошибки и дополнительную информацию
    """
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=partial(collate_fn, tokenizer=tokenizer)
    )
    
    model.eval()
    predictions = []
    targets = []
    dish_ids = []
    texts = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Подготовка данных
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device)
            }
            values = batch['value'].to(device)
            
            # Инференс
            logits = model(**inputs)
            predicted = logits.squeeze()
            
            # Сохраняем результаты
            predictions.extend(predicted.cpu().numpy())
            targets.extend(values.cpu().numpy())
            
            # Получаем дополнительную информацию из исходного датасета
            batch_start = len(predictions) - len(predicted)
            for i in range(len(predicted)):
                idx = batch_start + i
                if idx < len(dataset):
                    sample = dataset[idx]
                    dish_ids.append(sample['dish_id'])
                    texts.append(sample['text'])
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    errors = np.abs(predictions - targets)
    
    # Вычисляем MAE
    mae = np.mean(errors)
    
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean prediction: {np.mean(predictions):.2f}")
    print(f"Mean target: {np.mean(targets):.2f}")
    print(f"Max error: {np.max(errors):.2f}")
    print(f"Min error: {np.min(errors):.2f}")
    
    return {
        'predictions': predictions,
        'targets': targets,
        'errors': errors,
        'dish_ids': dish_ids,
        'texts': texts,
        'mae': mae
    }


def find_worst_predictions(results, top_k=10):
    """
    Находит худшие предсказания (с наибольшими ошибками)
    """
    worst_indices = np.argsort(results['errors'])[-top_k:][::-1]  # Сортируем по убыванию ошибки
    
    worst_predictions = []
    for idx in worst_indices:
        worst_predictions.append({
            'dish_id': results['dish_ids'][idx],
            'text': results['texts'][idx],
            'predicted': results['predictions'][idx],
            'target': results['targets'][idx],
            'error': results['errors'][idx],
            'dataset_idx': idx
        })
    
    print(f"\nTop {top_k} worst predictions:")
    print("-" * 80)
    for i, pred in enumerate(worst_predictions):
        print(f"{i+1}. Dish ID: {pred['dish_id']}")
        print(f"   Text: {pred['text'][:60]}...")
        print(f"   Predicted: {pred['predicted']:.2f} cal/100g")
        print(f"   Actual: {pred['target']:.2f} cal/100g")
        print(f"   Error: {pred['error']:.2f}")
        print("-" * 80)
    
    return worst_predictions


def evaluate_model(config, model_path, dataset_type="test", top_k_worst=10):
    """
    Полная оценка модели: загрузка, инференс, вычисление метрик, поиск худших предсказаний
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Загрузка модели
    model = load_model(config, model_path, device)
    
    # Подготовка датасета
    if dataset_type == "test":
        transforms = get_transforms(config, ds_type="test")
        dataset = MultimodalDataset(config, transforms, ds_type="test")
    elif dataset_type == "val":
        transforms = get_transforms(config, ds_type="val")
        dataset = MultimodalDataset(config, transforms, ds_type="val")
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    
    print(f"Evaluating on {dataset_type} dataset with {len(dataset)} samples")
    
    # Инференс и оценка
    results = inference_and_evaluate(model, dataset, config, device)
    
    # Поиск худших предсказаний
    worst_predictions = find_worst_predictions(results, top_k=top_k_worst)
    
    return results, worst_predictions
