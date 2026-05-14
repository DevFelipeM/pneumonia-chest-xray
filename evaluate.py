"""Avaliação do modelo e geração de gráficos/métricas."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_on_test(learner, test_files, output_dir: Path = Path("outputs")):
    """Avalia o modelo no conjunto de teste oficial.

    Gera matriz de confusão, métricas completas e salva tudo em disco.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab = list(learner.dls.vocab)

    print(f"\nAvaliando {len(test_files)} imagens do conjunto de teste...")
    test_dl = learner.dls.test_dl(test_files, with_labels=True)
    preds, targets = learner.get_preds(dl=test_dl)

    y_true = targets.numpy()
    y_pred = preds.argmax(dim=1).numpy()
    y_score = preds[:, _pos_index(vocab)].numpy()

    metrics = _compute_metrics(y_true, y_pred, y_score, vocab)
    _save_metrics(metrics, output_dir)
    _plot_confusion_matrix(y_true, y_pred, vocab, output_dir)
    _save_classification_report(y_true, y_pred, vocab, output_dir)

    return metrics


def _pos_index(vocab) -> int:
    for i, label in enumerate(vocab):
        if str(label).lower().startswith("pneumonia"):
            return i
    return len(vocab) - 1


def _compute_metrics(y_true, y_pred, y_score, vocab) -> dict:
    pos = _pos_index(vocab)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=pos)),
        "recall": float(recall_score(y_true, y_pred, pos_label=pos)),
        "f1_score": float(f1_score(y_true, y_pred, pos_label=pos)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "classes": [str(v) for v in vocab],
        "positive_class": str(vocab[pos]),
        "n_test_samples": int(len(y_true)),
    }
    return metrics


def _save_metrics(metrics: dict, output_dir: Path):
    path = output_dir / "metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Métricas salvas em: {path}")
    print("\n=== Resultados no conjunto de teste ===")
    for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
        print(f"  {key:>10s}: {metrics[key]:.4f}")


def _plot_confusion_matrix(y_true, y_pred, vocab, output_dir: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=vocab,
        yticklabels=vocab,
        ax=ax,
        cbar=False,
    )
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão — Conjunto de Teste")
    fig.tight_layout()

    path = output_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Matriz de confusão salva em: {path}")


def _save_classification_report(y_true, y_pred, vocab, output_dir: Path):
    report = classification_report(
        y_true, y_pred, target_names=[str(v) for v in vocab], digits=4
    )
    path = output_dir / "classification_report.txt"
    with open(path, "w") as f:
        f.write(report)
    print(f"Relatório de classificação salvo em: {path}")


def plot_training_curves(history: pd.DataFrame, output_dir: Path = Path("outputs")):
    """Gera gráficos de loss e acurácia por época."""
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(epochs, history["train_loss"], label="Treino", marker="o")
    axes[0].plot(epochs, history["valid_loss"], label="Validação", marker="s")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Curva de Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if "accuracy" in history.columns:
        axes[1].plot(
            epochs, history["accuracy"] * 100, label="Acurácia (val)",
            marker="o", color="green",
        )
        axes[1].set_xlabel("Época")
        axes[1].set_ylabel("Acurácia (%)")
        axes[1].set_title("Curva de Acurácia (Validação)")
        axes[1].set_ylim(0, 100)
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    fig.tight_layout()
    path = output_dir / "training_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Curvas de treino salvas em: {path}")
