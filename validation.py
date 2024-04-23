import torch
from model import DistillBERTClass
from torch import cuda
import constants
from transformers import DistilBertTokenizer
from torch.utils.data import DataLoader
from functions import data_ingestion, valid
from data_loader import Triage

device = 'cuda' if cuda.is_available() else 'cpu'

# creating pandas dataframe from the csv data
df = data_ingestion(constants.gdrive_link)

# defining tokenizer 
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')

test_dataset = df.reset_index(drop=True)

print("TEST Dataset: {}".format(test_dataset.shape))

testing_set = Triage(test_dataset, tokenizer, constants.MAX_LEN)

test_params = {'batch_size': constants.VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

testing_loader = DataLoader(testing_set, **test_params)

model = DistillBERTClass()
model.to(device)

# Load the saved model weights into the initialized model
model.load_state_dict(torch.load("model_state_dict.pth", map_location=torch.device('cpu')))

# Creating the loss function and optimizer
loss_function = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=constants.LEARNING_RATE)

# Make sure to set the model to evaluation mode after loading
model.eval()

print('This is the validation section to print the accuracy and see how it performs')
print('Here we are leveraging on the dataloader crearted for the validation dataset, the approcah is using more of pytorch')

acc = valid(model, loss_function, testing_loader, device)
print("Accuracy on test data = %0.2f%%" % acc)

