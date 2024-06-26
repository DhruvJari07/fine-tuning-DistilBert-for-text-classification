from transformers import DistilBertModel, DistilBertTokenizer


MODEL_NAME = "distilbert-base-uncased"
# Download and save the pre-trained model
model = DistilBertModel.from_pretrained(MODEL_NAME)
model.save_pretrained("./premodels")
# Download and save the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained("./premodels")
print("Model and tokenizer downloaded successfully!")