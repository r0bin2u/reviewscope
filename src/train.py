import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from src.data import load_config, load_review_data

def tokenize_data(dataset, tokenizer, max_length):
    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)
    return dataset.map(_tok, batched=True, remove_columns=["text"])
    
def main():
    cfg = load_config()
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    model = AutoModelForSequenceClassification.from_pretrained(cfg["model"]["name"], num_labels=cfg["model"]["num_labels"])
    
    train_ds, test_ds = load_review_data(cfg)
    train_ds = tokenize_data(train_ds, tokenizer, cfg["model"]["max_length"])
    test_ds = tokenize_data(test_ds, tokenizer, cfg["model"]["max_length"])
    train_ds.set_format("torch")
    train_ds.set_format("torch")
    test_ds.set_format("torch")

    args = TrainingArguments(output_dir=cfg["training"]["output_dir"], num_train_epochs=cfg["training"]["num_train_epochs"],
                             per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"], 
                             learning_rate=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"],
                             warmup_ratio=cfg["training"]["warmup_ratio"], eval_strategy="epoch", 
                             save_strategy="epoch", load_best_model_at_end=True, logging_steps=50)
    
    trainer = Trainer(model=model, train_dataset=train_ds, eval_dataset=test_ds, args=args)
    trainer.train()
    
    trainer.save_model(cfg["training"]["output_dir"])
    tokenizer.save_pretrained(cfg["training"]["output_dir"])
    print(f"Model and tokenizer saved to {cfg['training']['output_dir']}")
    
if __name__ == "__main__":
    main()



