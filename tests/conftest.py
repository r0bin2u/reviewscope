from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_classifier():
    """Create a mock ReviewClassifier that returns predictable results."""
    classifier = MagicMock()
    classifier.predict.return_value = ("positive", 0.9823)
    return classifier


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": MagicMock(),
        "attention_mask": MagicMock(),
    }
    return tokenizer
