"""Recupera o modelo a partir do checkpoint .pth e exporta o .pkl.

Útil quando o treino foi interrompido (Ctrl+C) depois que o
SaveModelCallback já gravou o melhor checkpoint, mas antes de
train_model concluir o export final.

Após rodar este script, basta executar `python main.py`: ele
detecta o .pkl em outputs/ (main.py:47) e pula direto para a
avaliação no conjunto de teste.
"""
from pathlib import Path

from data import build_dataloaders, download_dataset
from model import build_learner

CHECKPOINT_NAME = "pneumonia_resnet50"
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def main():
    print("=" * 60)
    print("Recuperação de checkpoint → exportação .pkl")
    print("=" * 60)

    print("\n[1/4] Resolvendo dataset (usa cache do kagglehub)...")
    root = download_dataset()

    print("\n[2/4] Reconstruindo DataLoaders (mesmo seed=42)...")
    dls = build_dataloaders(root)

    print("\n[3/4] Reconstruindo Learner ResNet50 e carregando pesos...")
    learner = build_learner(dls)
    # learner.load procura em learner.path/learner.model_dir/<name>.pth;
    # o checkpoint mora em <project>/models/, então forçamos path = project root.
    learner.path = PROJECT_ROOT

    checkpoint_path = PROJECT_ROOT / "models" / f"{CHECKPOINT_NAME}.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint_path}")
    print(f"Carregando pesos de: {checkpoint_path}")
    learner.load(CHECKPOINT_NAME)

    print("\n[4/4] Exportando modelo para .pkl...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    export_path = OUTPUT_DIR / f"{CHECKPOINT_NAME}.pkl"
    learner.export(export_path)
    print(f"Modelo exportado em: {export_path}")

    print("\nPronto. Agora rode: python main.py")


if __name__ == "__main__":
    main()
