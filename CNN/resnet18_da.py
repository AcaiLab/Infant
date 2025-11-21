import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.autograd import Function
from copy import deepcopy
from PIL import Image

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)

class ResNet18_DANN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        for p in base.parameters():
            p.requires_grad = False

        self.feature_extractor = nn.Sequential(
            *list(base.children())[:-1],
        )

        feat_dim = base.fc.in_features
        self.class_classifier = nn.Linear(feat_dim, num_classes)
        self.domain_classifier = nn.Linear(feat_dim, 2)

    def forward(self, x, alpha=0.0):
        feats = self.feature_extractor(x)
        feats = torch.flatten(feats, 1)

        class_logits = self.class_classifier(feats)

        reversed_feats = grad_reverse(feats, alpha)
        domain_logits = self.domain_classifier(reversed_feats)

        return class_logits, domain_logits

def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    DATA_ROOT = "data"
    ADULT_DIR = os.path.join(DATA_ROOT, "adult")
    INFANT_DIR = os.path.join(DATA_ROOT, "infant")

    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LR = 1e-4
    LAMBDA_DA = 0.5

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

    adult_datasets = {
        split: datasets.ImageFolder(
            root=os.path.join(ADULT_DIR, split),
            transform=data_transforms[split]
        )
        for split in splits
    }

    infant_datasets = {
        split: datasets.ImageFolder(
            root=os.path.join(INFANT_DIR, split),
            transform=data_transforms[split]
        )
        for split in splits
    }

    adult_classes = adult_datasets["train"].classes
    infant_classes = infant_datasets["train"].classes
    assert adult_classes == infant_classes, \
        f"Adult classes {adult_classes} and infant classes {infant_classes} differ!"

    class_names = adult_classes
    NUM_CLASSES = len(class_names)
    print("Classes:", class_names)
    print("NUM_CLASSES:", NUM_CLASSES)

    adult_train_loader = DataLoader(
        adult_datasets["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    infant_train_loader = DataLoader(
        infant_datasets["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    infant_val_loader = DataLoader(
        infant_datasets["val"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    infant_test_loader = DataLoader(
        infant_datasets["test"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    model = ResNet18_DANN(num_classes=NUM_CLASSES).to(device)
    print(model)

    criterion_class = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=1e-4
    )

    def evaluate_classification(model, dataloader, desc="Eval"):
        model.eval()
        running_corrects = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                class_logits, _ = model(inputs, alpha=0.0)
                _, preds = torch.max(class_logits, 1)

                running_corrects += torch.sum(preds == labels).item()
                total += labels.size(0)

        acc = running_corrects / total if total > 0 else 0.0
        print(f"{desc} Accuracy: {acc:.4f}")
        return acc

    def train_da(model, adult_loader, infant_loader, val_loader,
                 num_epochs=20, lambda_da=0.5):

        best_model_wts = deepcopy(model.state_dict())
        best_val_acc = 0.0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 20)

            model.train()

            running_loss = 0.0
            running_class_loss = 0.0
            running_domain_loss = 0.0
            running_corrects = 0
            total = 0

            for (adult_inputs, adult_labels), (infant_inputs, infant_labels) in zip(adult_loader, infant_loader):
                adult_inputs = adult_inputs.to(device)
                adult_labels = adult_labels.to(device)
                infant_inputs = infant_inputs.to(device)
                infant_labels = infant_labels.to(device)

                inputs = torch.cat([adult_inputs, infant_inputs], dim=0)
                class_labels = torch.cat([adult_labels, infant_labels], dim=0)

                domain_labels = torch.cat([
                    torch.zeros(adult_inputs.size(0), dtype=torch.long),
                    torch.ones(infant_inputs.size(0), dtype=torch.long)
                ], dim=0).to(device)

                optimizer.zero_grad()

                alpha = lambda_da

                class_logits, domain_logits = model(inputs, alpha=alpha)

                loss_class = criterion_class(class_logits, class_labels)
                loss_domain = criterion_domain(domain_logits, domain_labels)

                loss = loss_class + lambda_da * loss_domain
                loss.backward()
                optimizer.step()

                _, preds = torch.max(class_logits, 1)
                running_loss += loss.item() * inputs.size(0)
                running_class_loss += loss_class.item() * inputs.size(0)
                running_domain_loss += loss_domain.item() * inputs.size(0)
                running_corrects += torch.sum(preds == class_labels).item()
                total += class_labels.size(0)

            epoch_loss = running_loss / total
            epoch_class_loss = running_class_loss / total
            epoch_domain_loss = running_domain_loss / total
            epoch_acc = running_corrects / total

            print(
                f"Train Total Loss: {epoch_loss:.4f} | "
                f"Class Loss: {epoch_class_loss:.4f} | "
                f"Domain Loss: {epoch_domain_loss:.4f} | "
                f"Class Acc (adult+infant): {epoch_acc:.4f}"
            )

            val_acc = evaluate_classification(model, val_loader, desc="Infant val")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_wts = deepcopy(model.state_dict())

        print(f"\nTraining complete. Best infant val accuracy = {best_val_acc:.4f}")
        model.load_state_dict(best_model_wts)
        return model

    model = train_da(
        model,
        adult_train_loader,
        infant_train_loader,
        infant_val_loader,
        num_epochs=NUM_EPOCHS,
        lambda_da=LAMBDA_DA
    )

    print("\n=== Final evaluation on INFANT test set ===")
    test_acc = evaluate_classification(model, infant_test_loader, desc="Infant test")

    def predict_infant_image(model, img_path):
        model.eval()
        img = Image.open(img_path).convert("RGB")
        transform = data_transforms["test"]
        x = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            class_logits, _ = model(x, alpha=0.0)
            probs = torch.softmax(class_logits, dim=1)
            conf, pred = torch.max(probs, 1)

        label = class_names[pred.item()]
        print(f"Predicted: {label} (conf = {conf.item():.3f})")
        return label, conf.item()

if __name__ == "__main__":
    main()
