from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# load your fine-tuned model from local disk
model = DistilBertForSequenceClassification.from_pretrained("src/models/poker_distilbert")
tokenizer = DistilBertTokenizer.from_pretrained("src/models/poker_distilbert")

# push to HuggingFace Hub
# this creates a new model repo under your username
repo_name = "jinha1491/poker-distilbert"

model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)

print(f"Model uploaded to https://huggingface.co/{repo_name}")
