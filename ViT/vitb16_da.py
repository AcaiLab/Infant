import os
import math
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm
import timm

SOURCE_ROOT = "./data/adult"
TARGET_ROOT = "./data/infant"

IMG_SIZE = 224
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 20
LAMBDA_DA = 1.0

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

NUM_WORKERS = 0
CHECKPOINT = "vit_b16_dann_best.pth"

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

class ViTDANN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.encoder = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0
        )
        feat_dim = self.encoder.num_features

        self.class_head = nn.Linear(feat_dim, num_classes)

        self.domain_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )

    def forward(self, x, alpha=1.0):
        feats = self.encoder(x)
        class_logits = self.class_head(feats)

        rev_feats = grad_reverse(feats, alpha)
        domain_logits = self.domain_head(rev_feats)

        return class_logits, domain_logits


def compute_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).float().sum()
    return correct / labels.size(0)


def build_dataloaders():
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

    src_train = datasets.ImageFolder(
        root=os.path.join(SOURCE_ROOT, "train"),
        transform=train_transform
    )
    src_val = datasets.ImageFolder(
        root=os.path.join(SOURCE_ROOT, "val"),
        transform=val_transform
    )

    tgt_train = datasets.ImageFolder(
        root=os.path.join(TARGET_ROOT, "train"),
        transform=train_transform
    )
    tgt_val = datasets.ImageFolder(
        root=os.path.join(TARGET_ROOT, "val"),
        transform=val_transform
    )
    tgt_test = datasets.ImageFolder(
        root=os.path.join(TARGET_ROOT, "test"),
        transform=val_transform
    )

    print("Source classes:", src_train.classes)
    print("Target classes:", tgt_train.classes)

    pin_memory = (DEVICE == "cuda")

    src_train_loader = DataLoader(
        src_train, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=pin_memory
    )
    src_val_loader = DataLoader(
        src_val, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin_memory
    )
    tgt_train_loader = DataLoader(
        tgt_train, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=pin_memory
    )
    tgt_val_loader = DataLoader(
        tgt_val, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin_memory
    )
    tgt_test_loader = DataLoader(
        tgt_test, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin_memory
    )

    return (
        src_train_loader,
        src_val_loader,
        tgt_train_loader,
        tgt_val_loader,
        tgt_test_loader,
        len(src_train.classes),
    )


def dann_alpha(epoch, total_epochs, batch_idx, total_batches):
    p = float(epoch * total_batches + batch_idx) / (total_epochs * total_batches)
    return 2.0 / (1.0 + math.exp(-10 * p)) - 1.0


def train_one_epoch(model, src_loader, tgt_loader, optimizer,
                    cls_criterion, dom_criterion, epoch, total_epochs):
    model.train()
    total_src_loss = 0.0
    total_src_acc = 0.0
    n_src_batches = 0

    tgt_iter = iter(tgt_loader)
    total_batches = min(len(src_loader), len(tgt_loader))

    for batch_idx, (src_data, src_labels) in enumerate(
            tqdm(src_loader, desc=f"Epoch {epoch} [Train]", leave=False)):

        try:
            tgt_data, _ = next(tgt_iter)
        except StopIteration:
            tgt_iter = iter(tgt_loader)
            tgt_data, _ = next(tgt_iter)

        src_data = src_data.to(DEVICE)
        src_labels = src_labels.to(DEVICE)
        tgt_data = tgt_data.to(DEVICE)

        alpha = dann_alpha(epoch - 1, total_epochs, batch_idx, total_batches)

        src_class_logits, src_domain_logits = model(src_data, alpha=alpha)
        src_domain_labels = torch.zeros(src_data.size(0), dtype=torch.long, device=DEVICE)

        _, tgt_domain_logits = model(tgt_data, alpha=alpha)
        tgt_domain_labels = torch.ones(tgt_data.size(0), dtype=torch.long, device=DEVICE)

        cls_loss = cls_criterion(src_class_logits, src_labels)
        src_dom_loss = dom_criterion(src_domain_logits, src_domain_labels)
        tgt_dom_loss = dom_criterion(tgt_domain_logits, tgt_domain_labels)
        dom_loss = src_dom_loss + tgt_dom_loss

        loss = cls_loss + LAMBDA_DA * dom_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_src_loss += cls_loss.item()
        total_src_acc += compute_accuracy(src_class_logits, src_labels).item()
        n_src_batches += 1

    avg_src_loss = total_src_loss / n_src_batches
    avg_src_acc = total_src_acc / n_src_batches
    return avg_src_loss, avg_src_acc


@torch.no_grad()
def evaluate_classification(model, loader, cls_criterion, desc="Eval"):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for imgs, labels in tqdm(loader, desc=desc, leave=False):
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        class_logits, _ = model(imgs, alpha=0.0)
        loss = cls_criterion(class_logits, labels)
        acc = compute_accuracy(class_logits, labels)

        total_loss += loss.item()
        total_acc += acc.item()
        n_batches += 1

    return total_loss / n_batches, total_acc / n_batches


def main():
    print(f"Using device: {DEVICE}")

    (
        src_train_loader,
        src_val_loader,
        tgt_train_loader,
        tgt_val_loader,
        tgt_test_loader,
        num_classes,
    ) = build_dataloaders()

    model = ViTDANN(num_classes=num_classes).to(DEVICE)

    cls_criterion = nn.CrossEntropyLoss()
    dom_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_tgt_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        src_train_loss, src_train_acc = train_one_epoch(
            model,
            src_train_loader,
            tgt_train_loader,
            optimizer,
            cls_criterion,
            dom_criterion,
            epoch,
            EPOCHS,
        )

        src_val_loss, src_val_acc = evaluate_classification(
            model, src_val_loader, cls_criterion,
            desc=f"Epoch {epoch} [Src Val]"
        )

        tgt_val_loss, tgt_val_acc = evaluate_classification(
            model, tgt_val_loader, cls_criterion,
            desc=f"Epoch {epoch} [Tgt Val]"
        )

        print(f"""
        ============================================
        Epoch {epoch}/{EPOCHS}
        Source Train   - Loss: {src_train_loss:.4f}  Acc: {src_train_acc:.4f}
        Source Val     - Loss: {src_val_loss:.4f}    Acc: {src_val_acc:.4f}
        Target Val     - Loss: {tgt_val_loss:.4f}    Acc: {tgt_val_acc:.4f}
        ============================================
        """)

        if tgt_val_acc > best_tgt_val_acc:
            best_tgt_val_acc = tgt_val_acc
            torch.save(model.state_dict(), CHECKPOINT)
            print(f">>> Saved new best DA model to {CHECKPOINT} "
                  f"(Target Val Acc: {tgt_val_acc:.4f})")

    print("Training finished.")
    print(f"Best Target Val Accuracy: {best_tgt_val_acc:.4f}")

    print("\n[TEST] Loading best checkpoint and evaluating on target TEST set...")
    best_model = ViTDANN(num_classes=num_classes).to(DEVICE)
    best_model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))

    tgt_test_loss, tgt_test_acc = evaluate_classification(
        best_model, tgt_test_loader, cls_criterion, desc="Target Test"
    )

    print(f"""
    ============================================
    FINAL TARGET TEST PERFORMANCE
    Target Test Loss: {tgt_test_loss:.4f}
    Target Test Acc:  {tgt_test_acc:.4f}
    ============================================
    """)


if __name__ == "__main__":
    main()
