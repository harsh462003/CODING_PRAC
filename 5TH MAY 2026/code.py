"""
hyperparameter_study.py
=======================
EfficientNet-B0 Hyperparameter Impact Study on Fashion-MNIST
Author : Harshith Nagaraj  (251100610023)
Run    : python hyperparameter_study.py
Outputs: ./outputs/  (models / plots / results)
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os, sys, time, random, copy, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless – no display needed on GPU server
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from tqdm import tqdm

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device        : {device}")
if torch.cuda.is_available():
    print(f"GPU Name      : {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("WARNING: No GPU found – running on CPU. This will be very slow.")

# =============================================================================
# SECTION 1 – DIRECTORIES & CONFIG
# =============================================================================

BASE_OUT   = "./outputs"
MODEL_DIR  = f"{BASE_OUT}/models"
PLOT_DIR   = f"{BASE_OUT}/plots"
RESULT_DIR = f"{BASE_OUT}/results"
DATA_DIR   = "./data"

for d in [BASE_OUT, MODEL_DIR, PLOT_DIR, RESULT_DIR, DATA_DIR]:
    os.makedirs(d, exist_ok=True)

print("\nOutput directories:")
for d in [BASE_OUT, MODEL_DIR, PLOT_DIR, RESULT_DIR]:
    print(f"  {d}")

GLOBAL_SEED = 42

BASELINE_CONFIG = {
    "epochs"      : 5,
    "batch_size"  : 32,
    "lr"          : 0.001,
    "optimizer"   : "Adam",
    "activation"  : "ReLU",
    "padding"     : "none",
    "num_classes" : 10,
    "val_fraction": 0.1,
    "early_stop"  : False,
    "patience"    : 3,
    "use_gpu"     : True,
}

FMNIST_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

print("\nBaseline Config:")
for k, v in BASELINE_CONFIG.items():
    print(f"  {k:15s}: {v}")

# =============================================================================
# SECTION 2 – HELPER FUNCTIONS
# =============================================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(GLOBAL_SEED)


def get_activation(name: str) -> nn.Module:
    """Return a fresh activation module by name."""
    mapping = {
        "ReLU"     : nn.ReLU(),
        "LeakyReLU": nn.LeakyReLU(0.1),
        "GELU"     : nn.GELU(),
        "SiLU"     : nn.SiLU(),
    }
    if name not in mapping:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(mapping.keys())}")
    return mapping[name]


def get_optimizer(name: str, params, lr: float) -> optim.Optimizer:
    """
    Return an optimizer by name.
    params is materialised into a list first to avoid exhausting a generator
    (building a dict of all three constructors simultaneously would drain it).
    """
    param_list = list(params)
    if len(param_list) == 0:
        raise ValueError("get_optimizer received an empty parameter list.")
    if name == "Adam":
        return optim.Adam(param_list, lr=lr)
    elif name == "SGD":
        return optim.SGD(param_list, lr=lr, momentum=0.9)
    elif name == "RMSprop":
        return optim.RMSprop(param_list, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer '{name}'. Choose from ['Adam', 'SGD', 'RMSprop']")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_epoch_metrics(outputs, targets) -> float:
    """Compute accuracy (%) from lists of logit tensors and label tensors."""
    all_preds  = torch.cat(outputs).argmax(dim=1).numpy()
    all_labels = torch.cat(targets).numpy()
    return float((all_preds == all_labels).mean() * 100.0)


def evaluate_metrics(model, loader):
    """
    Full evaluation on a DataLoader.
    Returns: accuracy, precision, recall, f1, all_preds, all_labels
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs  = imgs.to(device)
            preds = model(imgs).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    acc  = float((np.array(all_preds) == np.array(all_labels)).mean() * 100.0)
    prec = float(precision_score(all_labels, all_preds, average="macro", zero_division=0) * 100)
    rec  = float(recall_score   (all_labels, all_preds, average="macro", zero_division=0) * 100)
    f1   = float(f1_score       (all_labels, all_preds, average="macro", zero_division=0) * 100)
    return acc, prec, rec, f1, all_preds, all_labels


def plot_curves(history: dict, title: str, save_path: str):
    """Save train/val loss and accuracy curves for one experiment."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=13)

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   marker="s")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve"); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(epochs, history["train_acc"], label="Train Acc", marker="o")
    axes[1].plot(epochs, history["val_acc"],   label="Val Acc",   marker="s")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy Curve"); axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_comparison(df: pd.DataFrame, x_col: str, y_col: str,
                    title: str, save_path: str, palette: str = "viridis"):
    """Save a bar-chart comparing one metric across hyperparameter values."""
    fig, ax = plt.subplots(figsize=(8, 5))
    df_sorted = df.sort_values(y_col, ascending=False).reset_index(drop=True)
    sns.barplot(data=df_sorted, x=x_col, y=y_col, palette=palette, ax=ax)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(x_col); ax.set_ylabel(y_col)
    for bar in ax.patches:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{bar.get_height():.2f}",
            ha="center", va="bottom", fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def save_results(records: list, csv_path: str) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"  Saved results → {csv_path}")
    return df


class EarlyStopping:
    def __init__(self, patience: int = 3, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss = None
        self.stop      = False

    def __call__(self, val_loss: float):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_loss = val_loss
            self.counter   = 0


print("\nHelper functions defined ✓")

# =============================================================================
# SECTION 3 – DATA
# =============================================================================

def build_transform(padding_strategy: str = "none") -> transforms.Compose:
    """
    Build preprocessing pipeline: grayscale 28×28 → RGB 224×224 (ImageNet stats).

    Fashion-MNIST is 1-channel. Grayscale(num_output_channels=3) replicates the
    channel across R, G, B so pretrained EfficientNet weights can be used.

    padding_strategy:
        "none"     – direct Resize to 224×224
        "constant" – zero-pad by 6px on each side (28→40), then Resize
        "reflect"  – reflection-pad by 6px on each side, then Resize
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    )
    to_rgb = transforms.Grayscale(num_output_channels=3)

    if padding_strategy == "none":
        return transforms.Compose([
            to_rgb,
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            normalize,
        ])
    elif padding_strategy == "constant":
        return transforms.Compose([
            to_rgb,
            transforms.Pad(6, fill=0, padding_mode="constant"),
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            normalize,
        ])
    elif padding_strategy == "reflect":
        return transforms.Compose([
            to_rgb,
            transforms.Pad(6, padding_mode="reflect"),
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        raise ValueError(f"Unknown padding strategy: '{padding_strategy}'")


def load_datasets(padding_strategy: str = "none",
                  val_fraction: float = 0.1,
                  seed: int = 42):
    tf = build_transform(padding_strategy)

    full_train = FashionMNIST(root=DATA_DIR, train=True,  download=True, transform=tf)
    test_set   = FashionMNIST(root=DATA_DIR, train=False, download=True, transform=tf)

    n_val   = int(len(full_train) * val_fraction)
    n_train = len(full_train) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [n_train, n_val], generator=generator)

    print(f"  Train samples : {len(train_set):,}")
    print(f"  Val samples   : {len(val_set):,}")
    print(f"  Test samples  : {len(test_set):,}")
    return train_set, val_set, test_set


def build_loaders(batch_size: int = 32,
                  padding_strategy: str = "none",
                  val_fraction: float = 0.1,
                  seed: int = 42,
                  num_workers: int = 4):          # increased from 2 for GPU server
    train_set, val_set, test_set = load_datasets(
        padding_strategy=padding_strategy,
        val_fraction=val_fraction,
        seed=seed,
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


# Quick verification
print("\nLoading Fashion-MNIST (padding=none) for sanity check…")
_tr, _va, _te = load_datasets()
print("Dataset loaded ✓")

# =============================================================================
# SECTION 4 – MODEL
# =============================================================================

def build_model(num_classes: int = 10,
                activation: str = "ReLU",
                freeze_backbone: bool = False) -> nn.Module:
    """
    Pretrained EfficientNet-B0 with a custom 2-layer classifier head.
    Only the classifier is replaced; the backbone (features) is kept intact.

    Original classifier: Sequential(Dropout(0.2), Linear(1280, 1000))
    New classifier     : Sequential(Dropout(0.2), Linear(1280, 256), Act, Linear(256, n_classes))
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model   = efficientnet_b0(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features   # 1280
    act         = get_activation(activation)

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 256),
        act,
        nn.Linear(256, num_classes),
    )
    return model.to(device)


# Sanity check
_m = build_model()
print(f"\nTrainable parameters : {count_parameters(_m):,}")
print(f"Classifier head      :\n{_m.classifier}")
print("Model builder defined ✓")
del _m

# =============================================================================
# SECTION 5 – TRAINING LOOP
# =============================================================================

def run_experiment(exp_name: str, config: dict) -> tuple:
    """
    Train one experiment, validate each epoch, test with best weights.
    Returns (metrics_dict, history_dict).
    """
    set_seed(GLOBAL_SEED)

    epochs       = config["epochs"]
    batch_size   = config["batch_size"]
    lr           = config["lr"]
    opt_name     = config["optimizer"]
    act_name     = config["activation"]
    padding      = config["padding"]
    num_classes  = config["num_classes"]
    val_fraction = config["val_fraction"]
    use_early    = config["early_stop"]
    patience     = config["patience"]

    print(f"\n{'='*60}")
    print(f"Experiment : {exp_name}")
    print(f"  lr={lr}, bs={batch_size}, opt={opt_name}, "
          f"act={act_name}, pad={padding}, epochs={epochs}")
    print(f"{'='*60}")

    train_loader, val_loader, test_loader = build_loaders(
        batch_size=batch_size,
        padding_strategy=padding,
        val_fraction=val_fraction,
        seed=GLOBAL_SEED,
    )

    model     = build_model(num_classes=num_classes, activation=act_name)
    optimizer = get_optimizer(opt_name, model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()

    if use_early:
        stopper = EarlyStopping(patience=patience)

    n_params     = count_parameters(model)
    best_val_acc = 0.0
    best_state   = None

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        epoch_outputs, epoch_targets = [], []

        for imgs, labels in tqdm(train_loader,
                                  desc=f"Epoch {epoch}/{epochs} [Train]",
                                  leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            epoch_outputs.append(logits.detach().cpu())
            epoch_targets.append(labels.cpu())

        train_loss = running_loss / len(train_loader.dataset)
        train_acc  = compute_epoch_metrics(epoch_outputs, epoch_targets)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        val_outputs, val_targets = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss   = criterion(logits, labels)
                val_loss_sum += loss.item() * imgs.size(0)
                val_outputs.append(logits.cpu())
                val_targets.append(labels.cpu())

        val_loss = val_loss_sum / len(val_loader.dataset)
        val_acc  = compute_epoch_metrics(val_outputs, val_targets)

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"  ].append(round(val_loss,   4))
        history["train_acc" ].append(round(train_acc,  4))
        history["val_acc"   ].append(round(val_acc,    4))

        print(f"  Epoch {epoch:2d} | "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.2f}%  |  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = copy.deepcopy(model.state_dict())

        if use_early:
            stopper(val_loss)
            if stopper.stop:
                print(f"  Early stopping triggered at epoch {epoch}.")
                break

    total_time = time.time() - start_time

    # ── Save best model ───────────────────────────────────────────────────────
    model_path = f"{MODEL_DIR}/{exp_name}.pth"
    torch.save(best_state, model_path)
    print(f"  Best model saved → {model_path}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    test_acc, prec, rec, f1, _, _ = evaluate_metrics(model, test_loader)
    print(f"  Test  acc={test_acc:.2f}%  prec={prec:.2f}%  rec={rec:.2f}%  f1={f1:.2f}%")

    # ── Learning curve ────────────────────────────────────────────────────────
    plot_curves(history,
                title=f"Learning Curves — {exp_name}",
                save_path=f"{PLOT_DIR}/{exp_name}_curves.png")

    metrics = {
        "experiment"      : exp_name,
        "lr"              : lr,
        "batch_size"      : batch_size,
        "optimizer"       : opt_name,
        "activation"      : act_name,
        "padding"         : padding,
        "epochs_run"      : len(history["train_loss"]),
        "best_val_acc"    : round(best_val_acc, 4),
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss"  : history["val_loss"  ][-1],
        "test_accuracy"   : round(test_acc, 4),
        "macro_precision" : round(prec,     4),
        "macro_recall"    : round(rec,      4),
        "macro_f1"        : round(f1,       4),
        "training_time_s" : round(total_time, 2),
        "trainable_params": n_params,
    }
    return metrics, history

# =============================================================================
# SECTION 6 – HYPERPARAMETER STUDIES
# =============================================================================

# ── 6a. Learning Rate ─────────────────────────────────────────────────────────
print("\n" + "#"*60)
print("# STUDY 1 : Learning Rate")
print("#"*60)

LR_VALUES = [0.01, 0.001, 0.0001]
lr_records = []
for lr in LR_VALUES:
    m, h = run_experiment(f"lr_{lr}", {**BASELINE_CONFIG, "lr": lr})
    lr_records.append(m)

lr_df = save_results(lr_records, f"{RESULT_DIR}/lr_study.csv")
lr_df["lr_label"] = lr_df["lr"].astype(str)
plot_comparison(lr_df, "lr_label", "test_accuracy",
                "Learning Rate vs Test Accuracy",
                f"{PLOT_DIR}/lr_test_accuracy.png")
plot_comparison(lr_df, "lr_label", "macro_f1",
                "Learning Rate vs Macro F1",
                f"{PLOT_DIR}/lr_macro_f1.png")
print("\nLearning Rate Study Results:")
print(lr_df[["experiment","lr","best_val_acc","test_accuracy","macro_f1","training_time_s"]].to_string(index=False))


# ── 6b. Batch Size ────────────────────────────────────────────────────────────
print("\n" + "#"*60)
print("# STUDY 2 : Batch Size")
print("#"*60)

BS_VALUES = [16, 32, 64]
bs_records = []
for bs in BS_VALUES:
    m, h = run_experiment(f"bs_{bs}", {**BASELINE_CONFIG, "batch_size": bs})
    bs_records.append(m)

bs_df = save_results(bs_records, f"{RESULT_DIR}/bs_study.csv")
bs_df["bs_label"] = bs_df["batch_size"].astype(str)
plot_comparison(bs_df, "bs_label", "test_accuracy",
                "Batch Size vs Test Accuracy",
                f"{PLOT_DIR}/bs_test_accuracy.png", palette="magma")
plot_comparison(bs_df, "bs_label", "macro_f1",
                "Batch Size vs Macro F1",
                f"{PLOT_DIR}/bs_macro_f1.png", palette="magma")
print("\nBatch Size Study Results:")
print(bs_df[["experiment","batch_size","best_val_acc","test_accuracy","macro_f1","training_time_s"]].to_string(index=False))


# ── 6c. Optimizer ─────────────────────────────────────────────────────────────
print("\n" + "#"*60)
print("# STUDY 3 : Optimizer")
print("#"*60)

OPT_VALUES = ["SGD", "Adam", "RMSprop"]
opt_records = []
for opt_name in OPT_VALUES:
    m, h = run_experiment(f"opt_{opt_name}", {**BASELINE_CONFIG, "optimizer": opt_name})
    opt_records.append(m)

opt_df = save_results(opt_records, f"{RESULT_DIR}/optimizer_study.csv")
plot_comparison(opt_df, "optimizer", "test_accuracy",
                "Optimizer vs Test Accuracy",
                f"{PLOT_DIR}/opt_test_accuracy.png", palette="coolwarm")
plot_comparison(opt_df, "optimizer", "macro_f1",
                "Optimizer vs Macro F1",
                f"{PLOT_DIR}/opt_macro_f1.png", palette="coolwarm")
print("\nOptimizer Study Results:")
print(opt_df[["experiment","optimizer","best_val_acc","test_accuracy","macro_f1","training_time_s"]].to_string(index=False))


# ── 6d. Epoch Count ───────────────────────────────────────────────────────────
print("\n" + "#"*60)
print("# STUDY 4 : Epoch Count")
print("#"*60)

EPOCH_VALUES = [3, 5, 8]
epoch_records = []
for ep in EPOCH_VALUES:
    m, h = run_experiment(f"epochs_{ep}", {**BASELINE_CONFIG, "epochs": ep})
    epoch_records.append(m)

epoch_df = save_results(epoch_records, f"{RESULT_DIR}/epoch_study.csv")
epoch_df["epoch_label"] = epoch_df["epochs_run"].astype(str)
plot_comparison(epoch_df, "epoch_label", "test_accuracy",
                "Epoch Count vs Test Accuracy",
                f"{PLOT_DIR}/epoch_test_accuracy.png", palette="cividis")
plot_comparison(epoch_df, "epoch_label", "macro_f1",
                "Epoch Count vs Macro F1",
                f"{PLOT_DIR}/epoch_macro_f1.png", palette="cividis")
print("\nEpoch Count Study Results:")
print(epoch_df[["experiment","epochs_run","best_val_acc","test_accuracy","macro_f1","training_time_s"]].to_string(index=False))


# ── 6e. Activation Function ───────────────────────────────────────────────────
print("\n" + "#"*60)
print("# STUDY 5 : Activation Function")
print("#"*60)

ACT_VALUES = ["ReLU", "LeakyReLU", "GELU", "SiLU"]
act_records = []
for act in ACT_VALUES:
    m, h = run_experiment(f"act_{act}", {**BASELINE_CONFIG, "activation": act})
    act_records.append(m)

act_df = save_results(act_records, f"{RESULT_DIR}/activation_study.csv")
plot_comparison(act_df, "activation", "test_accuracy",
                "Activation Function vs Test Accuracy",
                f"{PLOT_DIR}/act_test_accuracy.png", palette="plasma")
plot_comparison(act_df, "activation", "macro_f1",
                "Activation Function vs Macro F1",
                f"{PLOT_DIR}/act_macro_f1.png", palette="plasma")
print("\nActivation Study Results:")
print(act_df[["experiment","activation","best_val_acc","test_accuracy","macro_f1","training_time_s"]].to_string(index=False))


# ── 6f. Padding Strategy ──────────────────────────────────────────────────────
print("\n" + "#"*60)
print("# STUDY 6 : Padding Strategy")
print("#"*60)

PAD_VALUES = ["none", "constant", "reflect"]
pad_records = []
for pad in PAD_VALUES:
    m, h = run_experiment(f"pad_{pad}", {**BASELINE_CONFIG, "padding": pad})
    pad_records.append(m)

pad_df = save_results(pad_records, f"{RESULT_DIR}/padding_study.csv")
plot_comparison(pad_df, "padding", "test_accuracy",
                "Padding Strategy vs Test Accuracy",
                f"{PLOT_DIR}/pad_test_accuracy.png", palette="mako")
plot_comparison(pad_df, "padding", "macro_f1",
                "Padding Strategy vs Macro F1",
                f"{PLOT_DIR}/pad_macro_f1.png", palette="mako")
print("\nPadding Strategy Study Results:")
print(pad_df[["experiment","padding","best_val_acc","test_accuracy","macro_f1","training_time_s"]].to_string(index=False))

# =============================================================================
# SECTION 7 – MASTER RESULTS TABLE
# =============================================================================

print("\n" + "#"*60)
print("# MASTER RESULTS")
print("#"*60)

master_df = pd.concat([lr_df, bs_df, opt_df, epoch_df, act_df, pad_df],
                       ignore_index=True)
master_df = master_df.drop_duplicates(subset="experiment", keep="first")

master_csv = f"{RESULT_DIR}/master_results.csv"
master_df.to_csv(master_csv, index=False)
print(f"Master CSV saved → {master_csv}")

master_sorted = master_df.sort_values(
    ["best_val_acc", "macro_f1"], ascending=[False, False]
).reset_index(drop=True)

display_cols = [
    "experiment", "lr", "batch_size", "optimizer",
    "activation", "padding", "epochs_run",
    "best_val_acc", "test_accuracy", "macro_f1", "training_time_s",
]
print("\n── All Experiments Sorted by Validation Accuracy ──")
print(master_sorted[display_cols].to_string(index=False))

best_row = master_sorted.iloc[0]
print(f"\n🏆 Best Experiment  : {best_row['experiment']}")
print(f"   Val Accuracy     : {best_row['best_val_acc']:.2f}%")
print(f"   Test Accuracy    : {best_row['test_accuracy']:.2f}%")
print(f"   Macro F1         : {best_row['macro_f1']:.2f}%")

# =============================================================================
# SECTION 8 – BEST MODEL DEEP EVALUATION
# =============================================================================

print("\n" + "#"*60)
print("# BEST MODEL EVALUATION")
print("#"*60)

best_exp_name = master_sorted.iloc[0]["experiment"]
best_cfg_row  = master_sorted.iloc[0]

best_cfg = {
    "epochs"      : int(best_cfg_row["epochs_run"]),
    "batch_size"  : int(best_cfg_row["batch_size"]),
    "lr"          : float(best_cfg_row["lr"]),
    "optimizer"   : best_cfg_row["optimizer"],
    "activation"  : best_cfg_row["activation"],
    "padding"     : best_cfg_row["padding"],
    "num_classes" : BASELINE_CONFIG["num_classes"],
    "val_fraction": BASELINE_CONFIG["val_fraction"],
    "early_stop"  : BASELINE_CONFIG["early_stop"],
    "patience"    : BASELINE_CONFIG["patience"],
    "use_gpu"     : BASELINE_CONFIG["use_gpu"],
}

print(f"Loading best model: {best_exp_name}")
best_model = build_model(num_classes=best_cfg["num_classes"],
                          activation=best_cfg["activation"])
best_model.load_state_dict(
    torch.load(f"{MODEL_DIR}/{best_exp_name}.pth", map_location=device)
)
best_model.eval()

_, _, test_loader_best = build_loaders(
    batch_size=best_cfg["batch_size"],
    padding_strategy=best_cfg["padding"],
    val_fraction=best_cfg["val_fraction"],
    seed=GLOBAL_SEED,
)

test_acc, prec, rec, f1, all_preds, all_labels = evaluate_metrics(
    best_model, test_loader_best
)
print(f"\nFinal Test Metrics:")
print(f"  Accuracy  : {test_acc:.2f}%")
print(f"  Precision : {prec:.2f}%")
print(f"  Recall    : {rec:.2f}%")
print(f"  F1 Score  : {f1:.2f}%")

# Classification report
report = classification_report(all_labels, all_preds,
                                 target_names=FMNIST_CLASSES, digits=4)
print(f"\nClassification Report:\n{report}")

report_path = f"{RESULT_DIR}/best_model_classification_report.txt"
with open(report_path, "w") as fh:
    fh.write(f"Best Experiment: {best_exp_name}\n\n")
    fh.write(report)
print(f"Report saved → {report_path}")

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=FMNIST_CLASSES, yticklabels=FMNIST_CLASSES,
            linewidths=0.5, ax=ax)
ax.set_title(f"Confusion Matrix — {best_exp_name}", fontsize=14, pad=15)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label",      fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
cm_path = f"{PLOT_DIR}/best_model_confusion_matrix.png"
plt.savefig(cm_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"Confusion matrix saved → {cm_path}")

# =============================================================================
# SECTION 9 – FINAL SUMMARY
# =============================================================================

def best_in_group(df, rank_col="test_accuracy"):
    return df.sort_values(rank_col, ascending=False).iloc[0]

b_lr  = best_in_group(lr_df)
b_bs  = best_in_group(bs_df)
b_opt = best_in_group(opt_df)
b_ep  = best_in_group(epoch_df)
b_act = best_in_group(act_df)
b_pad = best_in_group(pad_df)

print("\n" + "="*65)
print("         FINAL HYPERPARAMETER STUDY SUMMARY")
print("="*65)
print(f"\n{'Group':<28} {'Best Value':<14} {'Test Acc':>9}  {'Macro F1':>9}")
print("-"*65)
print(f"{'Learning Rate':<28} {str(b_lr['lr']):<14} {b_lr['test_accuracy']:>8.2f}%  {b_lr['macro_f1']:>8.2f}%")
print(f"{'Batch Size':<28} {str(int(b_bs['batch_size'])):<14} {b_bs['test_accuracy']:>8.2f}%  {b_bs['macro_f1']:>8.2f}%")
print(f"{'Optimizer':<28} {str(b_opt['optimizer']):<14} {b_opt['test_accuracy']:>8.2f}%  {b_opt['macro_f1']:>8.2f}%")
print(f"{'Epoch Count':<28} {str(int(b_ep['epochs_run'])):<14} {b_ep['test_accuracy']:>8.2f}%  {b_ep['macro_f1']:>8.2f}%")
print(f"{'Activation Function':<28} {str(b_act['activation']):<14} {b_act['test_accuracy']:>8.2f}%  {b_act['macro_f1']:>8.2f}%")
print(f"{'Padding Strategy':<28} {str(b_pad['padding']):<14} {b_pad['test_accuracy']:>8.2f}%  {b_pad['macro_f1']:>8.2f}%")
print("-"*65)

overall = master_sorted.iloc[0]
print(f"\n{'OVERALL BEST EXPERIMENT':<28} {overall['experiment']}")
print(f"  lr={overall['lr']}, bs={int(overall['batch_size'])}, "
      f"opt={overall['optimizer']}, act={overall['activation']}, "
      f"pad={overall['padding']}, epochs={int(overall['epochs_run'])}")
print(f"  Test Accuracy : {overall['test_accuracy']:.2f}%")
print(f"  Macro F1      : {overall['macro_f1']:.2f}%")
print(f"  Val Accuracy  : {overall['best_val_acc']:.2f}%")
print("="*65)

print("\nAuto-generated Observations:")
lr_ranked  = lr_df.sort_values("test_accuracy", ascending=False)["lr"].tolist()
bs_ranked  = bs_df.sort_values("test_accuracy", ascending=False)["batch_size"].tolist()
opt_ranked = opt_df.sort_values("test_accuracy", ascending=False)["optimizer"].tolist()
ep_ranked  = epoch_df.sort_values("test_accuracy", ascending=False)["epochs_run"].tolist()
act_ranked = act_df.sort_values("test_accuracy", ascending=False)["activation"].tolist()
pad_ranked = pad_df.sort_values("test_accuracy", ascending=False)["padding"].tolist()

print(f"  • Best LR={lr_ranked[0]} outperformed LR={lr_ranked[-1]} — "
      f"too-high LR may cause instability, too-low underfits in few epochs.")
print(f"  • Batch size {int(bs_ranked[0])} showed best accuracy. "
      f"Smaller batches provide noisier but often better-generalizing gradients.")
print(f"  • {opt_ranked[0]} optimizer ranked first. "
      f"Adaptive optimizers typically converge faster on transfer learning tasks.")
print(f"  • {int(ep_ranked[0])} epochs gave best results. "
      f"More epochs generally help unless overfitting occurs.")
print(f"  • {act_ranked[0]} activation in the classifier head yielded highest accuracy.")
print(f"  • Padding strategy '{pad_ranked[0]}' worked best.")

print("\nAll outputs saved to:")
print(f"  Models   → {MODEL_DIR}/")
print(f"  Plots    → {PLOT_DIR}/")
print(f"  Results  → {RESULT_DIR}/")
print(f"  Master   → {master_csv}")
print("\nStudy complete ✓")
