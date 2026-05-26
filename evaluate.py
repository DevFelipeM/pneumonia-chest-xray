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
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


MAIN_THRESHOLD = "youden"


def evaluate_on_test(learner, test_files, output_dir: Path = Path("outputs")):
    """Avalia o modelo no conjunto de teste oficial com threshold tuning.

    Fluxo:
    1. Gera predições no conjunto de validação para calibrar 4 thresholds
       (default 0.5, Youden's J, F1 máximo, recall PNEUMONIA >= 99%).
    2. Aplica cada threshold no test set e reporta métricas.
    3. Figuras (matriz de confusão, ROC, relatório) usam o threshold "principal"
       definido em MAIN_THRESHOLD; ROC marca todos os operating points.

    IMPORTANTE: thresholds são escolhidos na VALIDAÇÃO e aplicados no TEST.
    Calibrar threshold no próprio test set é vazamento (overfit ao test).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab = list(learner.dls.vocab)
    pos = _pos_index(vocab)
    pos_class = str(vocab[pos])

    print("\nGerando predições no conjunto de validação para calibrar thresholds...")
    val_preds, val_targets = learner.get_preds()  # default: dls.valid
    if len(val_targets) == 0:
        raise RuntimeError(
            "dls.valid está vazio. Se o learner foi carregado de .pkl, "
            "reata os dataloaders antes de chamar evaluate_on_test "
            "(learner.dls = build_dataloaders(...))."
        )
    y_val_true = val_targets.numpy()
    y_val_score = val_preds[:, pos].numpy()

    thresholds = _find_thresholds(y_val_true, y_val_score, pos)
    print("\n=== Thresholds calculados na validação ===")
    for name, t in thresholds.items():
        print(f"  {name:<10s}: {t:.4f}")

    print(f"\nAvaliando {len(test_files)} imagens do conjunto de teste...")
    test_dl = learner.dls.test_dl(test_files, with_labels=True)
    preds, targets = learner.get_preds(dl=test_dl)
    y_true = targets.numpy()
    y_score = preds[:, pos].numpy()

    metrics = _compute_all_metrics(y_true, y_score, thresholds, pos, vocab,
                                   n_validation=len(y_val_true))
    _save_metrics(metrics, output_dir)

    main_t = thresholds[MAIN_THRESHOLD]
    y_pred_main = _predict_at_threshold(y_score, main_t, pos)
    _plot_confusion_matrix(
        y_true, y_pred_main, vocab, output_dir,
        threshold_name=MAIN_THRESHOLD, threshold_value=main_t,
        metrics=metrics["test_metrics_by_threshold"][MAIN_THRESHOLD],
        auc=metrics["auc"],
    )
    _plot_roc_curve(y_true, y_score, pos, pos_class, metrics["auc"],
                    thresholds, output_dir)
    _save_classification_report(y_true, y_pred_main, vocab, output_dir,
                                threshold_name=MAIN_THRESHOLD, threshold_value=main_t)

    return metrics


def _pos_index(vocab) -> int:
    for i, label in enumerate(vocab):
        if str(label).lower().startswith("pneumonia"):
            return i
    return len(vocab) - 1


def _find_thresholds(y_true, y_score, pos) -> dict:
    """Acha thresholds ótimos por 3 critérios sobre o conjunto de validação.

    - default: 0.5 (argmax tradicional)
    - youden: maximiza TPR - FPR (ponto mais distante da diagonal na ROC)
    - f1: maximiza F1 da classe positiva
    - recall_99: maior threshold tal que recall positivo >= 99%
                 (critério clínico, prioriza não perder doente)
    """
    y_true_pos = (y_true == pos).astype(int)

    fpr, tpr, roc_thr = roc_curve(y_true_pos, y_score)
    j = tpr - fpr
    youden_t = float(roc_thr[np.argmax(j)])

    precision, recall, pr_thr = precision_recall_curve(y_true_pos, y_score)
    # precision_recall_curve retorna N+1 pontos e N thresholds; descarta o último ponto
    f1s = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-10)
    f1_t = float(pr_thr[np.argmax(f1s)])

    valid = np.where(recall[:-1] >= 0.99)[0]
    if len(valid) > 0:
        recall_99_t = float(np.max(pr_thr[valid]))
    else:
        # Fallback: nem o menor threshold atinge 99% -> usa o menor disponível
        recall_99_t = float(np.min(pr_thr)) if len(pr_thr) > 0 else 0.0

    return {
        "default": 0.5,
        "youden": youden_t,
        "f1": f1_t,
        "recall_99": recall_99_t,
    }


def _predict_at_threshold(y_score, threshold: float, pos: int) -> np.ndarray:
    """Aplica threshold em y_score (= P(classe positiva)) e devolve índices de vocab."""
    pred_pos = y_score >= threshold
    return np.where(pred_pos, pos, 1 - pos).astype(int)


def _compute_all_metrics(y_true, y_score, thresholds, pos, vocab, n_validation: int) -> dict:
    """AUC global + métricas (acc/prec/rec/f1) por threshold."""
    y_true_pos = (y_true == pos).astype(int)
    auc = float(roc_auc_score(y_true_pos, y_score))

    by_thr = {}
    for name, t in thresholds.items():
        y_pred = _predict_at_threshold(y_score, t, pos)
        by_thr[name] = {
            "threshold": float(t),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, pos_label=pos, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, pos_label=pos, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, pos_label=pos, zero_division=0)),
        }

    return {
        "auc": auc,
        "classes": [str(v) for v in vocab],
        "positive_class": str(vocab[pos]),
        "n_test_samples": int(len(y_true)),
        "n_validation_samples": int(n_validation),
        "main_threshold": MAIN_THRESHOLD,
        "test_metrics_by_threshold": by_thr,
    }


def _save_metrics(metrics: dict, output_dir: Path):
    path = output_dir / "metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMétricas salvas em: {path}")

    print("\n=== Métricas no test set por threshold ===")
    print(f"  {'critério':<10s}  {'thresh':>7s}  {'acc':>7s}  {'prec':>7s}  {'rec':>7s}  {'f1':>7s}")
    print("  " + "-" * 54)
    for name, m in metrics["test_metrics_by_threshold"].items():
        print(f"  {name:<10s}  {m['threshold']:>7.4f}  "
              f"{m['accuracy']:>7.4f}  {m['precision']:>7.4f}  "
              f"{m['recall']:>7.4f}  {m['f1_score']:>7.4f}")
    print(f"\n  AUC: {metrics['auc']:.4f}  (independente do threshold)")
    print(f"  Threshold principal (figuras): {metrics['main_threshold']}")


def _plot_confusion_matrix(y_true, y_pred, vocab, output_dir: Path,
                           threshold_name: str | None = None,
                           threshold_value: float | None = None,
                           metrics: dict | None = None,
                           auc: float | None = None):
    cm = confusion_matrix(y_true, y_pred)
    pos = _pos_index(vocab)
    n_classes = len(vocab)

    row_totals = cm.sum(axis=1, keepdims=True)
    cm_pct = cm / np.where(row_totals == 0, 1, row_totals) * 100

    annot = np.empty_like(cm, dtype=object)
    for i in range(n_classes):
        for j in range(n_classes):
            tag = _cell_label(i, j, pos, n_classes)
            prefix = f"{tag}\n" if tag else ""
            annot[i, j] = f"{prefix}{cm[i, j]}\n({cm_pct[i, j]:.1f}%)"

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=vocab,
        yticklabels=vocab,
        ax=ax,
        cbar=False,
        annot_kws={"size": 12},
    )
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    title = "Matriz de Confusão — Conjunto de Teste"
    if threshold_name and threshold_value is not None:
        title += f"\nThreshold: {threshold_name} = {threshold_value:.4f}"
    ax.set_title(title)

    if metrics is not None:
        summary_parts = [
            f"Acurácia: {metrics['accuracy']:.4f}",
            f"Precisão: {metrics['precision']:.4f}",
            f"Recall: {metrics['recall']:.4f}",
            f"F1: {metrics['f1_score']:.4f}",
        ]
        if auc is not None:
            summary_parts.append(f"AUC: {auc:.4f}")
        summary = "  |  ".join(summary_parts)
        fig.text(
            0.5, 0.02, summary, ha="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="#999"),
        )
        fig.tight_layout(rect=[0, 0.08, 1, 1])
    else:
        fig.tight_layout()

    path = output_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Matriz de confusão salva em: {path}")


def _cell_label(i: int, j: int, pos: int, n_classes: int) -> str:
    """Rótulo TP/TN/FP/FN para o caso binário; vazio caso contrário."""
    if n_classes != 2:
        return ""
    if i == pos and j == pos:
        return "TP"
    if i == pos and j != pos:
        return "FN"
    if i != pos and j == pos:
        return "FP"
    return "TN"


def _plot_roc_curve(y_true, y_score, pos: int, pos_class: str, auc: float,
                    thresholds: dict, output_dir: Path):
    y_true_pos = (y_true == pos).astype(int)
    fpr, tpr, _ = roc_curve(y_true_pos, y_score)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--",
            label="Classificador aleatório")

    # Operating points: como cada threshold se posiciona na ROC do test set.
    style = {
        "default":   {"color": "#d62728", "marker": "o", "size": 100},
        "youden":    {"color": "#2ca02c", "marker": "*", "size": 220},
        "f1":        {"color": "#ff7f0e", "marker": "s", "size": 100},
        "recall_99": {"color": "#9467bd", "marker": "D", "size": 90},
    }
    for name, t in thresholds.items():
        y_pred = _predict_at_threshold(y_score, t, pos)
        tp = int(np.sum((y_pred == pos) & (y_true == pos)))
        fn = int(np.sum((y_pred != pos) & (y_true == pos)))
        fp = int(np.sum((y_pred == pos) & (y_true != pos)))
        tn = int(np.sum((y_pred != pos) & (y_true != pos)))
        fpr_pt = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr_pt = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        s = style[name]
        ax.scatter([fpr_pt], [tpr_pt], color=s["color"], marker=s["marker"],
                   s=s["size"], zorder=5, edgecolor="black", linewidth=0.6,
                   label=f"{name} (t={t:.3f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Taxa de Falso Positivo (FPR)")
    ax.set_ylabel("Taxa de Verdadeiro Positivo (TPR)")
    ax.set_title(f"Curva ROC — Classe positiva: {pos_class}")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = output_dir / "roc_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Curva ROC salva em: {path}")


def _save_classification_report(y_true, y_pred, vocab, output_dir: Path,
                                threshold_name: str | None = None,
                                threshold_value: float | None = None):
    report = classification_report(
        y_true, y_pred, target_names=[str(v) for v in vocab], digits=4
    )
    header = ""
    if threshold_name and threshold_value is not None:
        header = f"Threshold aplicado: {threshold_name} = {threshold_value:.4f}\n\n"
    path = output_dir / "classification_report.txt"
    with open(path, "w") as f:
        f.write(header + report)
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
