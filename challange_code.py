"""
CodeToAGI — Deep Learning Series EP15 Challenge
Compare Initializations: Bad Random vs Xavier vs He on a 10-layer MLP (MNIST)

Run:
    python ep15_init_comparison.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# ── Config ───────────────────────────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE  = 128
EPOCHS      = 20
LR          = 1e-3
HIDDEN      = 256
LAYERS      = 10          # total Linear layers (input → 8 hidden → output)
SEED        = 42

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Data ─────────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_ds = datasets.MNIST("./data", train=True,  download=True, transform=transform)
val_ds   = datasets.MNIST("./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ── Model factory ────────────────────────────────────────────────────────────
def make_mlp(activation="tanh"):
    """10-layer MLP: 784 → 256×8 → 10"""
    act = nn.Tanh() if activation == "tanh" else nn.ReLU()
    layers = []
    in_dim = 784
    for i in range(LAYERS - 1):
        layers += [nn.Linear(in_dim, HIDDEN), act]
        in_dim = HIDDEN
    layers.append(nn.Linear(HIDDEN, 10))
    return nn.Sequential(*layers)

# ── Initialization helpers ───────────────────────────────────────────────────
def init_bad(m):
    """Too-large Gaussian init (std=1.0) — classic failure case"""
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=1.0)
        nn.init.zeros_(m.bias)

def init_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

def init_he(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        nn.init.zeros_(m.bias)

# ── Train / Eval ─────────────────────────────────────────────────────────────
def train_one_epoch(model, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for x, y in train_loader:
        x, y = x.view(x.size(0), -1).to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

@torch.no_grad()
def evaluate(model, criterion):
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    for x, y in val_loader:
        x, y = x.view(x.size(0), -1).to(DEVICE), y.to(DEVICE)
        logits = model(x)
        total_loss += criterion(logits, y).item()
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += y.size(0)
    return total_loss / len(val_loader), 100.0 * correct / total

def run_experiment(name, model, init_fn):
    print(f"\n{'='*60}")
    print(f"  Experiment: {name}")
    print(f"{'='*60}")
    model.apply(init_fn)
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, optimizer, criterion)
        val_loss, val_acc = evaluate(model, criterion)
        best_acc = max(best_acc, val_acc)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}/{EPOCHS}  "
                  f"train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  "
                  f"val_acc={val_acc:.2f}%")
    elapsed = time.time() - start
    print(f"  → Final val accuracy: {val_acc:.2f}%  |  Best: {best_acc:.2f}%  |  {elapsed:.1f}s")
    return val_acc, best_acc

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Training {LAYERS}-layer MLP on MNIST for {EPOCHS} epochs\n")

    results = {}

    # 1. Bad random init (std=1.0) + Tanh
    model_bad = make_mlp(activation="tanh")
    acc_bad, best_bad = run_experiment("Bad Random (std=1.0) + Tanh", model_bad, init_bad)
    results["Bad Random"] = (acc_bad, best_bad)

    # 2. Xavier + Tanh
    model_xav = make_mlp(activation="tanh")
    acc_xav, best_xav = run_experiment("Xavier Uniform + Tanh", model_xav, init_xavier)
    results["Xavier"] = (acc_xav, best_xav)

    # 3. He / Kaiming + ReLU
    model_he = make_mlp(activation="relu")
    acc_he, best_he = run_experiment("He / Kaiming Uniform + ReLU", model_he, init_he)
    results["He"] = (acc_he, best_he)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  FINAL COMPARISON (validation accuracy)")
    print("="*60)
    for name, (final, best) in results.items():
        print(f"  {name:<30}  final={final:6.2f}%   best={best:6.2f}%")
    print("="*60)
    print("\nPost these three numbers in the YouTube comments!")
    print("Challenge complete.")
