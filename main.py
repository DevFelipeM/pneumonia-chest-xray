"""Pipeline completo: download → treino → avaliação.

Detecção de Pneumonia em imagens de raio-X torácico usando
transfer learning com ResNet50 (CNN profunda) via FastAI.

Uso:
    python main.py [--epochs 15] [--batch-size 32] [--lr 1e-3]
"""
import argparse
from pathlib import Path

from fastai.learner import load_learner

from data import build_dataloaders, download_dataset, get_test_files
from evaluate import evaluate_on_test, plot_training_curves
from model import build_learner
from train import train_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15,
                        help="Número de épocas de fine-tuning (padrão: 15)")
    parser.add_argument("--frozen-epochs", type=int, default=3,
                        help="Épocas treinando apenas a cabeça (padrão: 3)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Tamanho do batch (padrão: 32)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate base (padrão: 1e-3)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"),
                        help="Pasta para gráficos e métricas")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Detecção de Pneumonia — Pipeline de Treinamento")
    print("=" * 60)

    print("\n[1/5] Baixando dataset...")
    root = download_dataset()

    pkl_path = args.output_dir / "pneumonia_resnet50.pkl"
    if pkl_path.exists():
        print(f"\n[2-4/5] Modelo já treinado encontrado em {pkl_path}; pulando treino.")
        learner = load_learner(pkl_path)
    else:
        print("\n[2/5] Preparando DataLoaders...")
        dls = build_dataloaders(root, batch_size=args.batch_size, seed=args.seed)

        print("\n[3/5] Construindo modelo ResNet50 (transfer learning)...")
        learner = build_learner(dls)

        print("\n[4/5] Treinando o modelo...")
        history = train_model(
            learner,
            frozen_epochs=args.frozen_epochs,
            unfrozen_epochs=args.epochs,
            base_lr=args.lr,
            output_dir=args.output_dir,
        )
        plot_training_curves(history, args.output_dir)

    print("\n[5/5] Avaliando no conjunto de teste oficial...")
    test_files = get_test_files(root)
    evaluate_on_test(learner, test_files, args.output_dir)

    print("\nPipeline concluído.")


if __name__ == "__main__":
    main()
