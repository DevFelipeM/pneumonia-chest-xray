"""Construção do modelo CNN ResNet50 com transfer learning."""
from fastai.metrics import Precision, Recall, F1Score, accuracy
from fastai.vision.all import vision_learner
from torchvision.models import resnet50


def build_learner(dls, model_dir: str = "models"):
    """Cria um Learner FastAI com ResNet50 pré-treinada na ImageNet.

    A arquitetura é uma CNN profunda com conexões residuais (skip connections),
    permitindo treinar redes muito profundas sem degradação do gradiente.
    """
    learner = vision_learner(
        dls,
        resnet50,
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
