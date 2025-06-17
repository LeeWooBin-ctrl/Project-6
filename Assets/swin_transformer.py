import os
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.notebook import tqdm
from sklearn import metrics
from sklearn.model_selection import train_test_split  # new import

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from timm import create_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

# === Configuration ===
DATA_DIR = "F:\\CheXpert\\archive"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_EPOCHS = 10
LR = 1e-4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# === Full 14 CheXpert labels ===
label_columns = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
]
NUM_CLASSES = len(label_columns)

# === Data Preparation with Train/Test Split ===
def load_and_split(csv_path, test_size=0.15, random_state=42):
    df = pd.read_csv(csv_path)
    df['Path'] = df['Path'].apply(
        lambda x: os.path.normpath(f"{DATA_DIR}/{x.replace('CheXpert-v1.0-small/', '')}")
    )
    df[label_columns] = df[label_columns].fillna(0).astype(int)
    train_df, valid_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True)

# === Custom Dataset ===
class CheXpertDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(row['Path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        labels = row[label_columns].to_numpy(dtype=np.float32)
        return image, torch.from_numpy(labels)

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
transform_valid = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# === Loss Function ignoring -1 labels ===
def masked_bce_with_logits_loss(logits, targets):
    mask = targets != -1
    logits_m = logits[mask]
    targets_m = targets[mask]
    return F.binary_cross_entropy_with_logits(logits_m, targets_m)

# === Training Function with Accuracy ===
def train_model(model, train_loader, valid_loader, optimizer, scheduler, device, epochs=NUM_EPOCHS):
    train_losses, valid_losses = [], []
    train_accs, valid_accs = [], []
    tokens_per_sec_list = []

    scaler = GradScaler()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with autocast():
                logits = model(images)
                loss = masked_bce_with_logits_loss(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            total_loss += loss.item()

            preds = (torch.sigmoid(logits) > 0.5).float()
            mask = labels != -1
            correct += (preds[mask] == labels[mask]).sum().item()
            total += mask.sum().item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        acc = 100 * correct / total if total > 0 else 0
        train_losses.append(avg_loss)
        train_accs.append(acc)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        start = time.time()
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                with autocast():
                    logits = model(images)
                    loss = masked_bce_with_logits_loss(logits, labels)
                val_loss += loss.item()

                preds = (torch.sigmoid(logits) > 0.5).float()
                mask = labels != -1
                val_correct += (preds[mask] == labels[mask]).sum().item()
                val_total += mask.sum().item()
        elapsed = time.time() - start
        avg_val_loss = val_loss / len(valid_loader)
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        valid_losses.append(avg_val_loss)
        valid_accs.append(val_acc)
        tokens_per_sec_list.append(len(valid_loader.dataset) / elapsed)

        print(f"Train Loss: {avg_loss:.4f}, Train Acc: {acc:.2f}%")
        print(f"Valid Loss: {avg_val_loss:.4f}, Valid Acc: {val_acc:.2f}%, Speed: {tokens_per_sec_list[-1]:.2f} img/s")

    # Plot Loss and Accuracy
    epochs_range = range(1, epochs+1)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs_range, train_losses, 'o-', label='Train Loss')
    plt.plot(epochs_range, valid_losses, 's-', label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.title('Loss Curve')

    plt.subplot(1,2,2)
    plt.plot(epochs_range, train_accs, 'o-', label='Train Acc')
    plt.plot(epochs_range, valid_accs, 's-', label='Valid Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.title('Accuracy Curve')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load and split dataset
    df_train, df_valid = load_and_split(TRAIN_CSV, test_size=0.15)

    train_dataset = CheXpertDataset(df_train, transform=transform)
    valid_dataset = CheXpertDataset(df_valid, transform=transform_valid)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    train_model(model, train_loader, valid_loader, optimizer, scheduler, DEVICE)

    save_path = "D:\\projects\\Pycharm_python1\\CheXpert\\Weights\\chexpert_swin_14class.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.cpu().state_dict(), save_path)
    print(f"✅ Model saved to: {save_path}")
