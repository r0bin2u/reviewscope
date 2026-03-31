import pytest
import torch
from unittest.mock import patch, MagicMock
from src.model import ReviewClassifier, LABEL_MAP


class TestLabelMap:
    def test_label_map_has_three_entries(self):
        assert len(LABEL_MAP) == 3

    def test_label_map_values(self):
        assert LABEL_MAP[0] == "negative"
        assert LABEL_MAP[1] == "neutral"
        assert LABEL_MAP[2] == "positive"


class TestReviewClassifier:
    @patch("src.model.AutoModelForSequenceClassification")
    @patch("src.model.AutoTokenizer")
    def test_init_loads_model_and_tokenizer(self, mock_tok_cls, mock_model_cls):
        classifier = ReviewClassifier("fake-path", device="cpu")
        mock_tok_cls.from_pretrained.assert_called_once_with("fake-path")
        mock_model_cls.from_pretrained.assert_called_once_with("fake-path")

    @patch("src.model.AutoModelForSequenceClassification")
    @patch("src.model.AutoTokenizer")
    def test_init_moves_model_to_device(self, mock_tok_cls, mock_model_cls):
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model_cls.from_pretrained.return_value = mock_model
        classifier = ReviewClassifier("fake-path", device="cpu")
        mock_model.to.assert_called_once_with("cpu")
        mock_model.eval.assert_called_once()

    @patch("src.model.AutoModelForSequenceClassification")
    @patch("src.model.AutoTokenizer")
    def test_predict_returns_label_and_confidence(self, mock_tok_cls, mock_model_cls):
        # Setup mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer

        # Setup mock model - return logits where index 2 (positive) is highest
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[0.1, 0.2, 3.0]])
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.return_value = mock_output
        mock_model_cls.from_pretrained.return_value = mock_model

        classifier = ReviewClassifier("fake-path", device="cpu")
        label, confidence = classifier.predict("Great food!")

        assert label == "positive"
        assert 0.0 <= confidence <= 1.0

    @patch("src.model.AutoModelForSequenceClassification")
    @patch("src.model.AutoTokenizer")
    def test_predict_negative_sentiment(self, mock_tok_cls, mock_model_cls):
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer

        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[5.0, 0.1, 0.1]])  # index 0 (negative) highest
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.return_value = mock_output
        mock_model_cls.from_pretrained.return_value = mock_model

        classifier = ReviewClassifier("fake-path", device="cpu")
        label, confidence = classifier.predict("Terrible!")

        assert label == "negative"
        assert confidence > 0.9

    @patch("src.model.AutoModelForSequenceClassification")
    @patch("src.model.AutoTokenizer")
    def test_predict_neutral_sentiment(self, mock_tok_cls, mock_model_cls):
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1]]),
            "attention_mask": torch.tensor([[1]]),
        }
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer

        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[1.0, 5.0, 0.5]])  # index 1 (neutral) highest
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.return_value = mock_output
        mock_model_cls.from_pretrained.return_value = mock_model

        classifier = ReviewClassifier("fake-path", device="cpu")
        label, conf = classifier.predict("Test")
        assert label == "neutral"
