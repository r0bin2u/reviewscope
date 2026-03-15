import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer 


LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

class ReviewClassifier:
    def __init__(self, model_path, device=None):
        self.device  = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device).eval()
        
    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        idx = probs.argmax().item()
        return LABEL_MAP[idx], probs[idx].item()