import torch
from functions import predict
from transformers import DistilBertTokenizer
from model import DistillBERTClass
from torch import cuda
import constants

device = 'cuda' if cuda.is_available() else 'cpu'

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')

model = DistillBERTClass()
model.to(device)
# Load the saved model weights into the initialized model
model.load_state_dict(torch.load("model_state_dict.pth", map_location=torch.device('cpu')))

# Creating the loss function and optimizer
loss_function = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=constants.LEARNING_RATE)

# Make sure to set the model to evaluation mode after loading
model.eval()

# Example usage
sentence = "This is a random sentence to predict if this pipeline works or not. I believe it should work just fine."
prediction = predict(model, tokenizer, device, sentence)
print("Prediction:", prediction)