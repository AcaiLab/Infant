import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import timm

DATA_ROOT = "./data/infant/"

IMG_SIZE = 224
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 20

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

NUM_WORKERS = 0
CHECKPOINT = "vit_b16_noDA_classification.pth"


def compute_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).float().sum()
    return correct / labels.size(0)


def main():
    print(f"Using device: {DEVICE}")

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    train_dir = os.path.join(DATA_ROOT, "train")
    val_dir = os.path.join(DATA_ROOT, "val")

    train_dataset = datasets.ImageFolder(root=train_dir,
                                         transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir,
                                       transform=val_transform)

    print("Classes:", train_dataset.classes)

    pin_memory = (DEVICE == "cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    num_classes = len(train_dataset.classes)

    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=True,
        num_classes=num_classes,
    )
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val_acc = 0.0


    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        for images, labels in tqdm(train_loader,
                                   desc=f"Epoch {epoch} [Train]",
                                   leave=False):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += compute_accuracy(logits, labels).item()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():
            for images, labels in tqdm(val_loader,
                                       desc=f"Epoch {epoch} [Val]",
                                       leave=False):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                logits = model(images)
                loss = criterion(logits, labels)
                acc = compute_accuracy(logits, labels)

                val_loss += loss.item()
                val_acc += acc.item()

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        print(f"""
        ==========================
        Epoch {epoch}/{EPOCHS}
        Train Loss: {train_loss:.4f}
        Train Acc:  {train_acc:.4f}
        Val Loss:   {val_loss:.4f}
        Val Acc:    {val_acc:.4f}
        ==========================
        """)
    
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT)
            print(f">>> Saved new best model to {CHECKPOINT} "
                  f"(Val Acc: {val_acc:.4f})")

    print("Training complete.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
