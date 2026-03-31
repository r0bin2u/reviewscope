# ReviewScope

A multilingual review sentiment classifier powered by BERT. It fine-tunes [`bert-base-multilingual-cased`](https://huggingface.co/google-bert/bert-base-multilingual-cased) on the Yelp Review dataset and serves predictions through a FastAPI REST API.

## Features

- **Three-class sentiment classification** — negative, neutral, positive
- **Multilingual support** — uses multilingual BERT as the backbone
- **Training pipeline** — HuggingFace Trainer with configurable hyperparameters
- **Experiment tracking** — MLflow integration for logging parameters, metrics, and model artifacts
- **REST API** — FastAPI-based serving with `/predict` and `/health` endpoints
- **GPU acceleration** — automatic CUDA detection and utilization

## Project Structure

```
reviewscope/
├── configs/
│   └── config.yaml            # Model, training, and data configuration
├── models/
│   └── bert-sentiment/        # Trained model checkpoint (generated after training)
├── src/
│   ├── api/
│   │   ├── main.py            # FastAPI application entry point
│   │   ├── routes.py          # API route definitions
│   │   ├── schemas.py         # Request/response Pydantic schemas
│   │   └── dependencies.py    # Dependency injection (model loading)
│   ├── data.py                # Dataset loading and label mapping
│   ├── model.py               # ReviewClassifier inference wrapper
│   └── train.py               # Training pipeline
├── tests/                     # Test suite
├── configs/config.yaml        # Training and model configuration
├── requirements.txt           # Python dependencies
├── Dockerfile
├── docker-compose.yml
└── LICENSE                    # MIT License
```

## Prerequisites

- Python 3.13+
- (Optional) NVIDIA GPU with CUDA for accelerated training and inference

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/r0bin2u/reviewscope.git
cd reviewscope
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Train the model

```bash
python -m src.train
```

This will:
- Download the [Yelp Review Full](https://huggingface.co/datasets/yelp_review_full) dataset
- Fine-tune `bert-base-multilingual-cased` for 3 epochs
- Save the trained model and tokenizer to `models/bert-sentiment/`
- Log experiment metrics to MLflow

### 4. Start the API server

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 5. Make a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The food was absolutely amazing!"}'
```

**Response:**

```json
{
  "text": "The food was absolutely amazing!",
  "sentiment": "positive",
  "confidence": 0.9823,
  "model_version": "v1.0"
}
```

### 6. Health check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy"
}
```

## Configuration

All training and model parameters are defined in [`configs/config.yaml`](configs/config.yaml):

| Section    | Parameter                      | Default                          | Description                          |
|------------|--------------------------------|----------------------------------|--------------------------------------|
| `model`    | `name`                         | `bert-base-multilingual-cased`   | Pretrained model from HuggingFace    |
| `model`    | `num_labels`                   | `3`                              | Number of classification labels      |
| `model`    | `max_length`                   | `128`                            | Maximum token sequence length        |
| `training` | `per_device_train_batch_size`  | `32`                             | Batch size per device                |
| `training` | `learning_rate`                | `2e-5`                           | Learning rate                        |
| `training` | `num_train_epochs`             | `3`                              | Number of training epochs            |
| `training` | `warmup_ratio`                 | `0.1`                            | Warmup proportion of total steps     |
| `training` | `weight_decay`                 | `0.01`                           | Weight decay for regularization      |
| `training` | `output_dir`                   | `models/bert-sentiment`          | Directory to save trained model      |
| `data`     | `dataset`                      | `yelp_review_full`               | HuggingFace dataset name             |
| `data`     | `test_size`                    | `0.15`                           | Fraction of data used for evaluation |
| `data`     | `seed`                         | `42`                             | Random seed for reproducibility      |

## API Reference

### `POST /predict`

Classify the sentiment of a review text.

**Request body:**

| Field  | Type   | Constraints              | Description           |
|--------|--------|--------------------------|-----------------------|
| `text` | string | 1-512 characters         | The review text       |

**Response body:**

| Field           | Type   | Description                                  |
|-----------------|--------|----------------------------------------------|
| `text`          | string | The input text (echoed back)                 |
| `sentiment`     | string | One of `negative`, `neutral`, `positive`     |
| `confidence`    | float  | Model confidence score (0-1)                 |
| `model_version` | string | Model version identifier                     |

### `GET /health`

Returns `{"status": "healthy"}` if the service is running.

## Label Mapping

The Yelp Review dataset uses 1-5 star ratings. These are mapped to three sentiment classes:

| Stars | Label      | ID |
|-------|------------|----|
| 1-2   | negative   | 0  |
| 3     | neutral    | 1  |
| 4-5   | positive   | 2  |

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
