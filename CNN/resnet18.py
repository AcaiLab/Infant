import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from copy import deepcopy
from PIL import Image


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LR = 1e-4
    DATA_DIR = "data/infant" 

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]),
        "test": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]),
    }

    splits = ["train", "val", "test"]

    image_datasets = {
        split: datasets.ImageFolder(
            root=os.path.join(DATA_DIR, split),
            transform=data_transforms[split]
        )
        for split in splits
    }

    dataloaders = {
        split: DataLoader(
            image_datasets[split],
            batch_size=BATCH_SIZE,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=False
        )
        for split in splits
    }


    class_names = image_datasets["train"].classes
    NUM_CLASSES = len(class_names)
    print("Classes:", class_names)
    print("NUM_CLASSES:", NUM_CLASSES)


    def build_resnet18(num_classes: int):
        # ResNet-18, ImageNet-pretrained 
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        for p in model.parameters():
            p.requires_grad = False

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

        return model

    model = build_resnet18(NUM_CLASSES).to(device)
    print(model.fc)

    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=LR, weight_decay=1e-4)

    def train_model(model, dataloaders, criterion, optimizer, num_epochs=20):
        best_model_wts = deepcopy(model.state_dict())
        best_val_acc = 0.0

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 20)

            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0
                total = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels).item()
                    total += labels.size(0)

                epoch_loss = running_loss / total
                epoch_acc = running_corrects / total

                history[f"{phase}_loss"].append(epoch_loss)
                history[f"{phase}_acc"].append(epoch_acc)

                print(f"{phase} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

                if phase == "val" and epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    best_model_wts = deepcopy(model.state_dict())

        print(f"\nTraining complete. Best validation accuracy = {best_val_acc:.4f}")
        model.load_state_dict(best_model_wts)
        return model, history

    model, history = train_model(model, dataloaders, criterion, optimizer, num_epochs=NUM_EPOCHS)

    def evaluate(model, dataloader):
        model.eval()
        running_corrects = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                running_corrects += torch.sum(preds == labels).item()
                total += labels.size(0)

        acc = running_corrects / total if total > 0 else 0.0
        print(f"\nFinal Test Accuracy: {acc:.4f}")
        return acc

    test_acc = evaluate(model, dataloaders["test"])

    def predict_image(model, img_path):
        model.eval()
        img = Image.open(img_path).convert("RGB")
        transform = data_transforms["test"]
        x = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        label = class_names[pred.item()]
        print(f"Predicted: {label} (conf = {conf.item():.3f})")
        return label, conf.item()

if __name__ == "__main__":
    main()
