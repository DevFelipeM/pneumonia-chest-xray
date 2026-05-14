# Pneumonia Detection (Chest X-Ray)

Pipeline completo para classificacao de pneumonia em radiografias toracicas usando transfer learning com ResNet50. A arquitetura utilizada e uma CNN profunda com conexoes residuais (ResNet), treinada com FastAI sobre PyTorch.

## Visao geral

O fluxo principal executa as seguintes etapas:

1. Download do dataset via KaggleHub.
2. Preparacao de dados com DataBlock (resize, augmentations, normalizacao ImageNet).
3. Treinamento em duas fases (cabeca congelada e fine-tuning).
4. Avaliacao no conjunto de teste oficial com metricas e graficos.

## Requisitos

- Python 3.10+
- Dependencias listadas em requirements.txt

## Instalacao

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Como executar

```bash
python main.py --epochs 15 --batch-size 32 --lr 1e-3
```

Argumentos disponiveis:

- `--epochs`: numero de epocas de fine-tuning (padrao: 15)
- `--frozen-epochs`: epocas treinando apenas a cabeca (padrao: 3)
- `--batch-size`: tamanho do batch (padrao: 32)
- `--lr`: learning rate base (padrao: 1e-3)
- `--output-dir`: pasta de saida (padrao: outputs)
- `--seed`: seed para reproducao

## Saidas geradas

Os resultados sao salvos em `outputs/`:

- `training_history.csv`: historico por epoca
- `training_curves.png`: curvas de loss e acuracia
- `metrics.json`: metricas no teste
- `confusion_matrix.png`: matriz de confusao
- `classification_report.txt`: relatorio detalhado
- `pneumonia_resnet50.pkl`: modelo exportado

## Estrutura do projeto

- `data.py`: download e DataLoaders
- `model.py`: definicao do modelo CNN (ResNet50) e metricas
- `train.py`: loop de treinamento e exportacao
- `evaluate.py`: avaliacao e visualizacoes
- `main.py`: pipeline completo

## Observacoes

- O conjunto `val/` oficial do dataset e pequeno, por isso o treinamento utiliza um split aleatorio de 20% sobre train+val e preserva `test/` para avaliacao final.
- O modelo usa transfer learning com pesos da ImageNet para acelerar a convergencia.

## Possiveis melhorias

- Validacao estratificada ou k-fold para reduzir variancia.
- Ajuste de limiar para priorizar recall clinico.
- Calibracao de probabilidades e curvas ROC/PR.
- Registro de experimentos com MLflow ou Weights & Biases.
