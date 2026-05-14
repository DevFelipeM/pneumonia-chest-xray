"""Carregamento e preparação do dataset Chest X-Ray Pneumonia."""
from pathlib import Path

import kagglehub
from fastai.vision.all import (
    CategoryBlock,
    DataBlock,
    ImageBlock,
    Normalize,
    PILImage,
    Resize,
    aug_transforms,
    get_image_files,
    imagenet_stats,
    parent_label,
)


DATASET_SLUG = "paultimothymooney/chest-xray-pneumonia"
IMG_SIZE = 224
DEFAULT_BATCH_SIZE = 32


def download_dataset() -> Path:
    """Baixa o dataset via kagglehub e retorna o caminho da pasta `chest_xray`."""
    base_path = Path(kagglehub.dataset_download(DATASET_SLUG))
    print(f"Dataset baixado em: {base_path}")

    chest_xray = base_path / "chest_xray"
    if not chest_xray.exists():
        # fallback: algumas versões aninham mais um nível
        nested = base_path / "chest_xray" / "chest_xray"
        chest_xray = nested if nested.exists() else base_path

    print(f"Pasta raiz das imagens: {chest_xray}")
    return chest_xray


def _get_train_val_images(root: Path):
    """Combina as pastas train/ e val/ em um único conjunto.

    O `val/` oficial tem apenas 16 imagens — usamos um split aleatório
    de 20% para validação, mantendo `test/` intocado para avaliação final.
    """
    files = []
    for split in ("train", "val"):
        split_dir = root / split
        if split_dir.exists():
            files.extend(get_image_files(split_dir))
    return files


def build_dataloaders(root: Path, batch_size: int = DEFAULT_BATCH_SIZE, seed: int = 42):
    """Cria os DataLoaders de treino/validação a partir de train/ + val/."""
    files = _get_train_val_images(root)
    if not files:
        raise RuntimeError(f"Nenhuma imagem encontrada em {root}")

    print(f"Total de imagens (treino + val): {len(files)}")

    dblock = DataBlock(
        blocks=(ImageBlock(cls=PILImage), CategoryBlock),
        get_items=lambda _: files,
        get_y=parent_label,
        splitter=_random_splitter(seed),
        item_tfms=Resize(IMG_SIZE),
        batch_tfms=[
            *aug_transforms(
                size=IMG_SIZE,
                flip_vert=False,
                max_rotate=10.0,
                max_zoom=1.1,
                max_lighting=0.2,
                max_warp=0.1,
            ),
            Normalize.from_stats(*imagenet_stats),
        ],
    )

    dls = dblock.dataloaders(root, bs=batch_size)
    print(f"Classes: {dls.vocab}")
    print(f"Treino: {len(dls.train_ds)} | Validação: {len(dls.valid_ds)}")
    return dls


def _random_splitter(seed: int):
    from fastai.data.transforms import RandomSplitter

    return RandomSplitter(valid_pct=0.2, seed=seed)


def get_test_files(root: Path):
    """Retorna a lista de imagens do conjunto de teste oficial."""
    test_dir = root / "test"
    if not test_dir.exists():
        raise RuntimeError(f"Pasta de teste não encontrada em {test_dir}")
    return get_image_files(test_dir)
