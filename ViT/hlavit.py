import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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
CHECKPOINT = "hlavit_infant_noDA_classification.pth"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        x = self.proj(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
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
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
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
        q = self.attn_vector.unsqueeze(0).unsqueeze(1)  
        q = q.expand(B, 1, D)

        scores = torch.bmm(q, x.transpose(1, 2)) / (D ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)

        pooled = torch.bmm(attn_weights, x)
        pooled = pooled.squeeze(1)
        return pooled


class HLAViT(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=4,
        embed_dim=256,
        depth_global=4,
        depth_local=4,
        num_heads=8,
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=0.0,
        local_pool_factor=2,
    ):
        super().__init__()

        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.image_size = image_size
        self.patch_size = patch_size
        self.local_pool_factor = local_pool_factor

        grid_size = image_size // patch_size
        self.grid_size = grid_size

        num_patches_global = grid_size * grid_size

        assert grid_size % local_pool_factor == 0, \
            "grid_size must be divisible by local_pool_factor"
        local_grid = grid_size // local_pool_factor
        num_patches_local = local_grid * local_grid

        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )

        self.pos_embed_global = nn.Parameter(torch.zeros(1, num_patches_global, embed_dim))
        self.pos_embed_local = nn.Parameter(torch.zeros(1, num_patches_local, embed_dim))
        nn.init.trunc_normal_(self.pos_embed_global, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_local, std=0.02)

        self.global_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
            )
            for _ in range(depth_global)
        ])
        self.global_norm = nn.LayerNorm(embed_dim)
        self.global_pool = AttentivePooling(embed_dim)

        self.local_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
            )
            for _ in range(depth_local)
        ])
        self.local_norm = nn.LayerNorm(embed_dim)
        self.local_pool = AttentivePooling(embed_dim)

        self.head = nn.Linear(embed_dim * 2, num_classes)

        self.apply(self._init_weights)

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

    def _global_tokens(self, feat_map):
        B, D, H_, W_ = feat_map.shape
        x = feat_map.view(B, D, H_ * W_)
        x = x.transpose(1, 2)
        x = x + self.pos_embed_global
        return x

    def _local_tokens(self, feat_map):
        B, D, H_, W_ = feat_map.shape
        f = self.local_pool_factor
        pooled = F.avg_pool2d(
            feat_map,
            kernel_size=f,
            stride=f
        )

        _, _, H_l, W_l = pooled.shape
        x = pooled.view(B, D, H_l * W_l)
        x = x.transpose(1, 2)
        x = x + self.pos_embed_local
        return x

    def forward(self, x):
        feat_map = self.patch_embed(x)

        g_tokens = self._global_tokens(feat_map)
        for blk in self.global_blocks:
            g_tokens = blk(g_tokens)
        g_tokens = self.global_norm(g_tokens)
        g_feat = self.global_pool(g_tokens)

        l_tokens = self._local_tokens(feat_map)
        for blk in self.local_blocks:
            l_tokens = blk(l_tokens)
        l_tokens = self.local_norm(l_tokens)
        l_feat = self.local_pool(l_tokens)

        fused = torch.cat([g_feat, l_feat], dim=1)  
        logits = self.head(fused)                 

        return logits

def compute_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).float().sum()
    return correct / labels.size(0)

def main():
    set_seed(42)
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

    model = HLAViT(
        image_size=IMG_SIZE,
        patch_size=16,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=256,
        depth_global=4,
        depth_local=4,
        num_heads=8,
        mlp_ratio=4.0,
        drop=0.1,
        attn_drop=0.1,
        local_pool_factor=2,
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
            print(f"Saved new best model to {CHECKPOINT} "
                  f"(Val Acc: {val_acc:.4f})")

    print("Training complete.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
