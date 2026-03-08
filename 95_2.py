# train_cifar10.py
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets, models

# -----------------------
# Config / Hyperparams
# -----------------------
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
NUM_WORKERS = 4
NUM_EPOCHS = 100          # you can run 30-50 for quick exam runs; 100+ for final
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
NUM_CLASSES = 10
IMG_SIZE = 32
MIXUP_ALPHA = 0.8        # set to 0 to disable mixup
LABEL_SMOOTHING = 0.1   # set to 0 to disable label smoothing
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# -----------------------
# Reproducibility
# -----------------------
def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

set_seed()

# -----------------------
# Data: transforms + loaders
# -----------------------
train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616]),
])

# For validation/test, no random transforms
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616]),
])

train_dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=train_transforms)
val_dataset   = datasets.CIFAR10(root="data", train=False, download=True, transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# -----------------------
# Model: ResNet18 adapted for CIFAR
# (replace first conv 7x7 with 3x3, remove early maxpool)
# -----------------------
def get_resnet18_cifar(pretrained=False, num_classes=NUM_CLASSES):
    model = models.resnet18(pretrained=pretrained)
    # modify first conv layer: original is 7x7 stride2, change to 3x3 stride1
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()  # remove the 3x3 maxpool
    # final fc
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

model = get_resnet18_cifar(pretrained=False).to(DEVICE)

# If you have multiple GPUs:
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

# -----------------------
# Loss function with label smoothing (optional)
# -----------------------
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.0):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, target):
        logprobs = self.log_softmax(x)
        nll_loss = -logprobs.gather(dim=1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

if LABEL_SMOOTHING > 0.0:
    criterion = LabelSmoothingCrossEntropy(smoothing=LABEL_SMOOTHING)
else:
    criterion = nn.CrossEntropyLoss()

# -----------------------
# MixUp helper (optional)
# -----------------------
def mixup_data(x, y, alpha=MIXUP_ALPHA):
    if alpha <= 0:
        return x, y, None, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

import numpy as np

# -----------------------
# Optimizer + Scheduler
# - SGD with momentum works very well on CIFAR
# - CosineAnnealingLR or OneCycleLR are both good; here's cosine
# -----------------------
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# -----------------------
# Training & Validation loops
# -----------------------
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k.mul_(100.0 / batch_size)).item())
        return res  # list of percentages

def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total = 0
    for i, (images, targets) in enumerate(train_loader):
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        # optional mixup
        if MIXUP_ALPHA > 0:
            images, targets_a, targets_b, lam = mixup_data(images, targets, alpha=MIXUP_ALPHA)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        acc1 = accuracy(outputs, targets, topk=(1,))[0] if MIXUP_ALPHA == 0 else None
        # For mixup we don't get meaningful single-label acc inside train; skip
        total += images.size(0)

    epoch_loss = running_loss / total
    return epoch_loss

def validate(epoch):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100.0 * correct / total
    val_loss = val_loss / total
    return val_loss, acc

# -----------------------
# Main training loop: checkpoint best model
# -----------------------
best_acc = 0.0
start_time = time.time()
for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()
    train_loss = train_one_epoch(epoch)
    val_loss, val_acc = validate(epoch)
    scheduler.step()

    is_best = val_acc > best_acc
    if is_best:
        best_acc = val_acc
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "acc": best_acc,
        }, CHECKPOINT_DIR / "best_resnet18_cifar10.pth")

    print(f"Epoch [{epoch}/{NUM_EPOCHS}]  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.2f}%  best={best_acc:.2f}%  time={time.time()-t0:.1f}s")

total_time = time.time() - start_time
print("Training finished. Best val acc: %.2f%%. Total time: %.1fs" % (best_acc, total_time))