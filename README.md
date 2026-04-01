# ReviewScope

Fine-tune multilingual BERT on the Yelp Review dataset, then serve sentiment predictions through a REST API.

The backbone is [`bert-base-multilingual-cased`](https://huggingface.co/google-bert/bert-base-multilingual-cased), which means it handles English, Chinese, Japanese, and many other languages out of the box. Reviews are classified into three categories — negative, neutral, and positive.

## Project Structure

```
reviewscope/
├── configs/
│   └── config.yaml          # hyperparameters, dataset, model config
├── src/
│   ├── api/
│   │   ├── main.py          # FastAPI app
│   │   ├── routes.py        # /predict and /health endpoints
│   │   ├── schemas.py       # request/response models
│   │   └── dependencies.py  # model dependency injection
│   ├── data.py              # dataset loading & label mapping
│   ├── model.py             # inference wrapper
│   └── train.py             # training pipeline with MLflow tracking
├── tests/                   # unit tests
├── .github/workflows/       # CI/CD (lint, test, Docker build & push)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── pyproject.toml
```

## Setup

```bash
git clone https://github.com/r0bin2u/reviewscope.git
cd reviewscope
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

GPU is optional but recommended — the code auto-detects CUDA.

## Train

```bash
python -m src.train
```

Downloads the [Yelp Review Full](https://huggingface.co/datasets/yelp_review_full) dataset (~650k reviews), fine-tunes BERT for 3 epochs, and saves the checkpoint to `models/bert-sentiment/`. Training metrics are logged to MLflow — run `mlflow ui` afterwards to browse them.

The Yelp dataset uses 1-5 star ratings, mapped to three classes:

| Stars | Label    |
|-------|----------|
| 1-2   | negative |
| 3     | neutral  |
| 4-5   | positive |

## Serve

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Or with Docker:

```bash
docker compose up --build
```

Note: the model checkpoint is mounted as a volume (`./models:/app/models`), not baked into the image. Train first, then start the container.

## API

**POST /predict**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The food was absolutely amazing!"}'
```

```json
{
  "text": "The food was absolutely amazing!",
  "sentiment": "positive",
  "confidence": 0.9875,
  "model_version": "v1.0"
}
```

The `text` field accepts 1-512 characters.

**GET /health**

```bash
curl http://localhost:8000/health
```

Returns `{"status": "healthy"}`.

## Configuration

All parameters live in [`configs/config.yaml`](configs/config.yaml):

| Parameter | Default | What it does |
|-----------|---------|-------------|
| `model.name` | `bert-base-multilingual-cased` | HuggingFace model |
| `model.num_labels` | `3` | output classes |
| `model.max_length` | `128` | max token length |
| `training.per_device_train_batch_size` | `32` | batch size |
| `training.learning_rate` | `2e-5` | learning rate |
| `training.num_train_epochs` | `3` | epochs |
| `training.warmup_ratio` | `0.1` | LR warmup |
| `training.weight_decay` | `0.01` | regularization |
| `data.dataset` | `yelp_review_full` | dataset name |
| `data.test_size` | `0.15` | eval split ratio |
| `data.seed` | `42` | random seed |

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## CI/CD

Push or PR to `main` triggers CI (ruff lint + pytest). Tagging `v*` triggers CD (build Docker image and push to GHCR).

```bash
git tag v1.0.0
git push origin v1.0.0
```

## License

MIT
