import numpy as np
import matplotlib.pyplot as plt


def denormalize_image(image_tensor, mean, std):
    """Convert normalized tensor image back to [0, 255] numpy array"""
    image = image_tensor.clone().numpy()
    for c in range(image.shape[0]):
        image[c] = image[c] * std[c] + mean[c]
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return np.transpose(image, (1, 2, 0))  # [H, W, C]


def visualize_dataset(dataset, num_images=8, rows=2, cols=4):
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    mean = dataset.image_cfg.mean
    std = dataset.image_cfg.std

    indices = np.random.choice(len(dataset), size=num_images, replace=False).tolist()

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        image = denormalize_image(sample['image'], mean, std)
        text = sample['text']
        value = sample['value']
        dish_id = sample['dish_id']

        ax = axes[i // cols, i % cols]
        ax.imshow(image)
        ax.set_title(f"ID: {dish_id}\n{value} cal/100g\n{text[:40]}...", fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def visualize_worst_predictions(dataset, worst_predictions, num_images=8, rows=2, cols=4):
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    mean = dataset.image_cfg.mean
    std = dataset.image_cfg.std
    
    num_to_show = min(num_images, len(worst_predictions))
    
    for i in range(num_to_show):
        pred_info = worst_predictions[i]
        
        dataset_idx = None
        for idx in range(len(dataset)):
            if dataset[idx]['dish_id'] == pred_info['dish_id']:
                dataset_idx = idx
                break
        
        if dataset_idx is not None:
            sample = dataset[dataset_idx]
            image = denormalize_image(sample['image'], mean, std)
            
            ax = axes[i // cols, i % cols] if rows > 1 else axes[i]
            ax.imshow(image)
            
            title = (f"ID: {pred_info['dish_id']}\n"
                    f"Predicted: {pred_info['predicted']:.1f} cal/100g\n"
                    f"Actual: {pred_info['target']:.1f} cal/100g\n"
                    f"Error: {pred_info['error']:.1f}\n"
                    f"{pred_info['text'][:30]}...")
            
            ax.set_title(title, fontsize=9)
            ax.axis('off')
        else:
            # Если не нашли изображение, показываем пустую ячейку
            ax = axes[i // cols, i % cols] if rows > 1 else axes[i]
            ax.text(0.5, 0.5, f"Image not found\nDish ID: {pred_info['dish_id']}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    
    # Заполняем оставшиеся ячейки пустотой
    for i in range(num_to_show, rows * cols):
        ax = axes[i // cols, i % cols] if rows > 1 else axes[i]
        ax.axis('off')
    
    plt.suptitle('Worst Predictions (Highest Errors)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()