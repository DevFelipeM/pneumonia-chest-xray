"""Loop de treinamento do modelo de detecção de pneumonia."""
from pathlib import Path

import pandas as pd
from fastai.callback.tracker import SaveModelCallback


def train_model(
    learner,
    frozen_epochs: int = 3,
    unfrozen_epochs: int = 15,
    base_lr: float = 1e-3,
    output_dir: Path = Path("outputs"),
    model_name: str = "pneumonia_resnet50",
):
    """Treina o modelo em duas fases:

    1. **Cabeça apenas** (camadas convolucionais congeladas): ajusta o
       classificador às novas classes.
    2. **Fine-tuning** (todas as camadas): refina os pesos pré-treinados
       para o domínio das radiografias torácicas.

    Retorna um DataFrame com o histórico de cada época.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    save_cb = SaveModelCallback(
        monitor="valid_loss",
        fname=model_name,
        with_opt=False,
    )

    print("\n=== Fase 1: treinando a cabeça (camadas congeladas) ===")
    learner.fit_one_cycle(frozen_epochs, base_lr, cbs=[save_cb])

    print("\n=== Fase 2: fine-tuning (todas as camadas) ===")
    learner.unfreeze()
    learner.fit_one_cycle(
        unfrozen_epochs,
        slice(base_lr / 100, base_lr / 10),
        cbs=[save_cb],
    )

    history = _extract_history(learner)
    history_path = output_dir / "training_history.csv"
    history.to_csv(history_path, index=False)
    print(f"\nHistórico de treino salvo em: {history_path}")

    learner.export(output_dir / f"{model_name}.pkl")
    print(f"Modelo exportado em: {output_dir / f'{model_name}.pkl'}")

    return history


def _extract_history(learner) -> pd.DataFrame:
    """Extrai o histórico do recorder do Learner como DataFrame."""
    recorder = learner.recorder
    columns = ["epoch", "train_loss", "valid_loss"] + [
        m.name for m in recorder.metrics
    ]
    rows = []
    for i, values in enumerate(recorder.values):
        rows.append([i] + [float(v) for v in values])
    return pd.DataFrame(rows, columns=columns)
