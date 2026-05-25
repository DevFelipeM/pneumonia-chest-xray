"""Construção do modelo CNN ResNet50 com transfer learning."""
from collections import Counter

import torch
from fastai.losses import CrossEntropyLossFlat
from fastai.metrics import Precision, Recall, F1Score, accuracy
from fastai.torch_core import default_device
from fastai.vision.all import parent_label, vision_learner
from torchvision.models import resnet50


def build_learner(dls, model_dir: str = "models", use_class_weights: bool = True):
    """Cria um Learner FastAI com ResNet50 pré-treinada na ImageNet.

    A arquitetura é uma CNN profunda com conexões residuais (skip connections),
    permitindo treinar redes muito profundas sem degradação do gradiente.

    Quando `use_class_weights=True` (padrão), a CrossEntropyLoss recebe pesos
    inversamente proporcionais à frequência de cada classe no conjunto de treino
    (estilo `sklearn` 'balanced'), compensando o desbalanceamento ~3:1
    pneumonia:normal do dataset Kaggle.
    """
    loss_func = None
    if use_class_weights:
        weights = _compute_class_weights(dls)
        print(f"Class weights ({list(dls.vocab)}): {weights.tolist()}")
        loss_func = CrossEntropyLossFlat(weight=weights.to(default_device()))

    learner = vision_learner(
        dls,
        resnet50,
        loss_func=loss_func,
        metrics=[
            accuracy,
            Precision(average="binary", pos_label=_pos_label_index(dls)),
            Recall(average="binary", pos_label=_pos_label_index(dls)),
            F1Score(average="binary", pos_label=_pos_label_index(dls)),
        ],
        model_dir=model_dir,
        pretrained=True,
    )
    return learner


def _pos_label_index(dls) -> int:
    """Índice da classe positiva (PNEUMONIA) no vocabulário."""
    vocab = list(dls.vocab)
    for i, label in enumerate(vocab):
        if str(label).lower().startswith("pneumonia"):
            return i
    # fallback: última classe
    return len(vocab) - 1


def _compute_class_weights(dls) -> torch.Tensor:
    """Pesos inversamente proporcionais à frequência das classes no treino.

    Fórmula 'balanced' do sklearn: w_i = n_total / (n_classes * freq_i).
    Lê os rótulos direto dos paths (sem carregar imagens) via `parent_label`.
    Ordem do tensor segue `dls.vocab`.
    """
    train_items = dls.train.items
    labels = [str(parent_label(p)) for p in train_items]
    counts = Counter(labels)
    vocab = [str(v) for v in dls.vocab]
    n_total = sum(counts.values())
    n_classes = len(vocab)
    weights = [n_total / (n_classes * counts[c]) for c in vocab]
    return torch.tensor(weights, dtype=torch.float32)
