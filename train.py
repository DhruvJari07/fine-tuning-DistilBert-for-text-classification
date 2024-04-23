# Importing the libraries needed
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertModel, DistilBertTokenizer
from data_loader import Triage
from functions import train, data_ingestion
import constants
from model import DistillBERTClass

#Setting up the device for GPU usage

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

# creating pandas dataframe from the csv data
df = data_ingestion(constants.gdrive_link)

# defining tokenizer 
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')

# Creating the dataset and dataloader for the neural network
train_dataset=df.sample(frac=0.8,random_state=200)
test_dataset=df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

print("FULL Dataset: {}".format(df.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = Triage(train_dataset, tokenizer, constants.MAX_LEN)
testing_set = Triage(test_dataset, tokenizer, constants.MAX_LEN)

train_params = {'batch_size': constants.TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': constants.VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

model = DistillBERTClass()
model.to(device)

# Creating the loss function and optimizer
loss_function = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=constants.LEARNING_RATE)

for epoch in range(constants.EPOCHS):
    train(model, loss_function, optimizer, training_loader, epoch, device)

# Save the model's state dictionary
torch.save(model.state_dict(), 'model_sample.pth')
