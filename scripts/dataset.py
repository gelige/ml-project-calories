import re
import numpy as np
import pandas as pd
import torch
import timm
import albumentations as A

from torch.utils.data import Dataset
from transformers import AutoTokenizer
from PIL import Image
from sklearn.model_selection import train_test_split

class MultimodalDataset(Dataset):
    def __init__(self, config, transforms, ds_type="train"):
        self.config = config

        # словарь ингредиентов
        ingredients_df = pd.read_csv(config.INGREDIENTS_PATH)
        self.ingredients = dict(zip(ingredients_df["id"], ingredients_df["ingr"]))

        # загружаем блюда и фильтруем по типу датасета (train или test)
        dishes_df = pd.read_csv(config.DISHES_PATH)

        # выбираем подходящий датасет
        if ds_type == "test":
            self.df = dishes_df[dishes_df["split"] == "test"].reset_index(drop=True)
        else:
            train_df = dishes_df[dishes_df["split"] == "train"]
            train_idx, val_idx = train_test_split(
                train_df.index,
                test_size=0.2,
                random_state=config.SEED,
                shuffle=True,
            )
            if ds_type == "train":
                self.df = train_df.loc[train_idx].copy().reset_index(drop=True)
            elif ds_type == "val":
                self.df = train_df.loc[val_idx].copy().reset_index(drop=True)
            else:
                raise ValueError(f"Unknown ds_type: {ds_type!r}")

        self.image_cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # калории на 100г
        value = self.df.loc[idx, "total_calories"] / self.df.loc[idx, "total_mass"] * 100

        # список ингредиентов в текстовом формате
        ingredient_ids = [int(re.search(r"\d+$", part).group()) for part in self.df.loc[idx, "ingredients"].split(";")]
        text = ", ".join([self.ingredients[ingredient_id] for ingredient_id in ingredient_ids])

        # изображение
        dish_id = self.df.loc[idx, "dish_id"]
        try:
            image = Image.open(f"{self.config.IMAGES_DIR}/{dish_id}/rgb.png").convert('RGB')
        except:
            image = torch.randint(0, 255, (*self.image_cfg.input_size[1:], self.image_cfg.input_size[0])).to(torch.float32)

        image = self.transforms(image=np.array(image))["image"]

        return {"value": value, "image": image, "text": text, "dish_id": dish_id}


def collate_fn(batch, tokenizer):
    texts = [item["text"] for item in batch]
    images = torch.stack([item["image"] for item in batch])
    value = torch.FloatTensor([item["value"] for item in batch])

    tokenized_input = tokenizer(texts,
                                return_tensors="pt",
                                padding="max_length",
                                truncation=True)

    return {
        "value": value,
        "image": images,
        "input_ids": tokenized_input["input_ids"],
        "attention_mask": tokenized_input["attention_mask"]
    }


def get_transforms(config, ds_type="train"):
    cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)

    if ds_type == "train":
        transforms = A.Compose(
            [
                A.Resize(height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
                A.Affine(scale=(0.9, 1.1),
                         rotate=(-45, 45),
                         translate_percent=(-0.1, 0.1),
                         shear=(-10, 10),
                         fill=0,
                         p=0.8),
                A.CoarseDropout(
                    num_holes_range=(2, 8),
                    hole_height_range=(int(0.07 * cfg.input_size[1]),
                                       int(0.15 * cfg.input_size[1])),
                    hole_width_range=(int(0.1 * cfg.input_size[2]),
                                      int(0.15 * cfg.input_size[2])),
                    fill=0,
                    p=0.5),
                A.ColorJitter(brightness=0.2,
                              contrast=0.2,
                              saturation=0.2,
                              hue=0.1,
                              p=0.7),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0)
            ],
            seed=config.SEED,
        )
    else:
        transforms = A.Compose(
            [
                A.Resize(height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0)
            ],
            seed=config.SEED,
        )

    return transforms
