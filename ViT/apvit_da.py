import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

DATA_ROOT = "./data"
SRC_DOMAIN = "adult"
TGT_DOMAIN = "infant"

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
CHECKPOINT = "apvit_DANN_adult2infant.pth"


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=256, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, D, H', W')
        x = x.flatten(2)  # (B, D, N)
        x = x.transpose(1, 2)  # (B, N, D)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,  # input: (B, N, D)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        # x: (B, N, D)
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)  # (B, N, D)
        x = x + attn_out

        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)
        return x


class AttentivePooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn_vector = nn.Parameter(torch.randn(embed_dim))

    def forward(self, x):
        B, N, D = x.shape
        q = self.attn_vector.unsqueeze(0).unsqueeze(1)  # (1, 1, D)
        q = q.expand(B, 1, D)                           # (B, 1, D)

        scores = torch.bmm(q, x.transpose(1, 2)) / (D ** 0.5)  # (B, 1, N)
        attn_weights = F.softmax(scores, dim=-1)               # (B, 1, N)

        pooled = torch.bmm(attn_weights, x)  # (B, 1, D)
        pooled = pooled.squeeze(1)           # (B, D)
        return pooled


class APViT(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=4,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=0.0,
    ):
        super().__init__()

        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        num_patches = (image_size // patch_size) ** 2

        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.att_pool = AttentivePooling(embed_dim=embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self.apply(self._init_weights)
        self.embed_dim = embed_dim

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_features(self, x):
        x = self.patch_embed(x)      
        x = x + self.pos_embed       
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)             
        pooled = self.att_pool(x)    
        return pooled

    def forward(self, x):
        feat = self.forward_features(x)
        logits = self.head(feat)
        return logits


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)


class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)


def compute_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).float().sum()
    return correct / labels.size(0)


def dann_alpha(epoch, total_epochs, batch_idx, num_batches):
    p = float(epoch - 1 + batch_idx / num_batches) / float(total_epochs)
    return 2.0 / (1.0 + np.exp(-10 * p)) - 1.0


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

    src_train_dir = os.path.join(DATA_ROOT, SRC_DOMAIN, "train")
    tgt_train_dir = os.path.join(DATA_ROOT, TGT_DOMAIN, "train")
    tgt_val_dir = os.path.join(DATA_ROOT, TGT_DOMAIN, "val")

    src_train_dataset = datasets.ImageFolder(root=src_train_dir,
                                             transform=train_transform)
    tgt_train_dataset = datasets.ImageFolder(root=tgt_train_dir,
                                             transform=train_transform)
    tgt_val_dataset = datasets.ImageFolder(root=tgt_val_dir,
                                           transform=val_transform)

    print("Source classes (adult):", src_train_dataset.classes)
    print("Target classes (infant):", tgt_train_dataset.classes)

    if src_train_dataset.classes != tgt_train_dataset.classes:
        print("Warning: source and target class names differ. "
              "Label mapping might be inconsistent.")

    pin_memory = (DEVICE == "cuda")

    src_train_loader = DataLoader(
        src_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    tgt_train_loader = DataLoader(
        tgt_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    tgt_val_loader = DataLoader(
        tgt_val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    num_classes = len(src_train_dataset.classes)

    feature_extractor = APViT(
        image_size=IMG_SIZE,
        patch_size=16,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        drop=0.1,
        attn_drop=0.1,
    ).to(DEVICE)

    domain_discriminator = DomainDiscriminator(in_dim=feature_extractor.embed_dim).to(DEVICE)

    cls_criterion = nn.CrossEntropyLoss()
    dom_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        list(feature_extractor.parameters()) + list(domain_discriminator.parameters()),
        lr=LR
    )

    best_val_acc = 0.0

    src_iter_len = len(src_train_loader)
    tgt_iter_len = len(tgt_train_loader)
    num_batches = min(src_iter_len, tgt_iter_len)

    for epoch in range(1, EPOCHS + 1):
        feature_extractor.train()
        domain_discriminator.train()

        train_cls_loss = 0.0
        train_dom_loss = 0.0
        train_cls_acc = 0.0

        src_iter = iter(src_train_loader)
        tgt_iter = iter(tgt_train_loader)

        for batch_idx in tqdm(range(num_batches),
                              desc=f"Epoch {epoch} [Train]",
                              leave=False):
            try:
                src_images, src_labels = next(src_iter)
            except StopIteration:
                src_iter = iter(src_train_loader)
                src_images, src_labels = next(src_iter)

            try:
                tgt_images, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_train_loader)
                tgt_images, _ = next(tgt_iter)

            src_images = src_images.to(DEVICE)
            src_labels = src_labels.to(DEVICE)
            tgt_images = tgt_images.to(DEVICE)

            optimizer.zero_grad()

            src_features = feature_extractor.forward_features(src_images)
            src_logits = feature_extractor.head(src_features)
            cls_loss = cls_criterion(src_logits, src_labels)
            cls_acc = compute_accuracy(src_logits, src_labels)

            tgt_features = feature_extractor.forward_features(tgt_images)

            features_all = torch.cat([src_features, tgt_features], dim=0)

            alpha = dann_alpha(epoch, EPOCHS, batch_idx, num_batches)
            features_rev = grad_reverse(features_all, alpha)

            dom_logits = domain_discriminator(features_rev)

            dom_labels_src = torch.zeros(src_features.size(0), dtype=torch.long, device=DEVICE)
            dom_labels_tgt = torch.ones(tgt_features.size(0), dtype=torch.long, device=DEVICE)
            dom_labels_all = torch.cat([dom_labels_src, dom_labels_tgt], dim=0)

            dom_loss = dom_criterion(dom_logits, dom_labels_all)

            loss = cls_loss + LAMBDA_DA * dom_loss
            loss.backward()
            optimizer.step()

            train_cls_loss += cls_loss.item()
            train_dom_loss += dom_loss.item()
            train_cls_acc += cls_acc.item()

        train_cls_loss /= num_batches
        train_dom_loss /= num_batches
        train_cls_acc /= num_batches

        feature_extractor.eval()
        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():
            for images, labels in tqdm(tgt_val_loader,
                                       desc=f"Epoch {epoch} [Val - Target]",
                                       leave=False):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                logits = feature_extractor(images)
                loss = cls_criterion(logits, labels)
                acc = compute_accuracy(logits, labels)

                val_loss += loss.item()
                val_acc += acc.item()

        val_loss /= len(tgt_val_loader)
        val_acc /= len(tgt_val_loader)

        print(f"""
        ==========================
        Epoch {epoch}/{EPOCHS}
        Train Src Cls Loss: {train_cls_loss:.4f}
        Train Dom Loss:     {train_dom_loss:.4f}
        Train Src Cls Acc:  {train_cls_acc:.4f}
        Val (Target) Loss:  {val_loss:.4f}
        Val (Target) Acc:   {val_acc:.4f}
        ==========================
        """)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "feature_extractor": feature_extractor.state_dict(),
                "domain_discriminator": domain_discriminator.state_dict(),
                "val_acc": best_val_acc,
                "epoch": epoch,
            }, CHECKPOINT)
            print(f"Saved new best model to {CHECKPOINT} "
                  f"(Val Target Acc: {val_acc:.4f})")

    print("Training complete.")
    print(f"Best target-val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
