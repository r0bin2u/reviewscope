from functools import lru_cache
from src.model import ReviewClassifier


@lru_cache(maxsize=1)
def get_model():
    return ReviewClassifier(model_path="models/bert-sentiment")