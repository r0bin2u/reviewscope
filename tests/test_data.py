from unittest.mock import MagicMock, patch

from src.data import load_config, load_review_data, star_to_label


class TestStarToLabel:
    def test_star_0_is_negative(self):
        assert star_to_label(0) == 0

    def test_star_1_is_negative(self):
        assert star_to_label(1) == 0

    def test_star_2_is_neutral(self):
        assert star_to_label(2) == 1

    def test_star_3_is_positive(self):
        assert star_to_label(3) == 2

    def test_star_4_is_positive(self):
        assert star_to_label(4) == 2

    def test_star_5_is_positive(self):
        assert star_to_label(5) == 2


class TestLoadConfig:
    @patch("builtins.open", create=True)
    @patch("src.data.yaml.safe_load")
    def test_load_config_returns_dict(self, mock_yaml, mock_open):
        mock_yaml.return_value = {"model": {"name": "bert"}}
        result = load_config("fake.yaml")
        assert result == {"model": {"name": "bert"}}

    @patch("builtins.open", create=True)
    @patch("src.data.yaml.safe_load")
    def test_load_config_default_path(self, mock_yaml, mock_open):
        mock_yaml.return_value = {}
        load_config()
        mock_open.assert_called_once_with("configs/config.yaml")


class TestLoadReviewData:
    @patch("src.data.load_dataset")
    def test_load_review_data_returns_train_test(self, mock_load_dataset):
        mock_ds = MagicMock()
        mock_mapped = MagicMock()
        mock_ds.map.return_value = mock_mapped
        mock_split = {"train": MagicMock(), "test": MagicMock()}
        mock_mapped.train_test_split.return_value = mock_split
        mock_load_dataset.return_value = mock_ds

        cfg = {"data": {"dataset": "yelp_review_full", "test_size": 0.15, "seed": 42}}
        train, test = load_review_data(cfg)

        mock_load_dataset.assert_called_once_with("yelp_review_full", split="train")
        mock_mapped.train_test_split.assert_called_once_with(test_size=0.15, seed=42)
        assert train == mock_split["train"]
        assert test == mock_split["test"]
