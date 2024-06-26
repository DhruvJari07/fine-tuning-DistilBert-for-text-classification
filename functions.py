import os
import gdown
import pandas as pd
import torch

# Defining the training function on the 80% of the dataset for tuning the distilbert model
def data_ingestion(gdrive_link):
    '''
    Fetch data from the url
    '''

    try: 
        dataset_url = gdrive_link
        download_dir = "artifacts/data.csv"
        os.makedirs("artifacts", exist_ok=True)

        file_id = dataset_url.split("/")[-2]
        prefix = 'https://drive.google.com/uc?/export=download&id='
        gdown.download(prefix+file_id,download_dir)
        df = pd.read_csv("artifacts/data.csv")

    except:
        raise Exception
    
    return df
    


def train(model, loss_function, optimizer, training_loader, epoch, device):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    for _,data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)
        outputs = model(ids, mask)
        loss = loss_function(outputs.squeeze(1), targets)  # Squeeze the output to match the shape of targets
        tr_loss += loss.item()
        # Apply a sigmoid activation to the outputs to obtain probabilities
        probabilities = torch.sigmoid(outputs)
        # Convert probabilities to binary predictions based on a threshold (e.g., 0.5)
        predictions = (probabilities > 0.5).float()
        n_correct += torch.sum(predictions.squeeze(1) == targets).item()

        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)

        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")

    return


def valid(model, loss_function, testing_loader, device):
    model.eval()
    tr_loss = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    n_correct = 0
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)

            outputs = model(ids, mask)
            loss = loss_function(outputs.squeeze(1), targets)  # Squeeze the output to match the shape of targets
            tr_loss += loss.item()
            # Apply a sigmoid activation to the outputs to obtain probabilities
            probabilities = torch.sigmoid(outputs)
            # Convert probabilities to binary predictions based on a threshold (e.g., 0.5)
            predictions = (probabilities > 0.5).float()
            n_correct += torch.sum(predictions.squeeze(1) == targets).item()

            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)

    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Validation Loss Epoch: {epoch_loss}")
    print(f"Validation Accuracy Epoch: {epoch_accu}")

    return epoch_accu


def prediction(model, tokenizer, device, sentence):
    # Tokenize the input sentence
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True).to(device)
    #print("inputs loaded")
    #print(f"inputs: {inputs}")
    #print("forward pass")
    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**inputs)
    #print(f"outputs: {outputs}")
    # Get the predicted probabilities
    probabilities = torch.sigmoid(outputs)
    #print(f"probabilities: {probabilities}")
    # Convert probabilities to binary predictions
    predictions = (probabilities > 0.5).int().squeeze()
    #print(f"proba is {probabilities}, pred is {predictions}")
    return predictions.item()