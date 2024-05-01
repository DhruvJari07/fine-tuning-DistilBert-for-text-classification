import json
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from torch import cuda


# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model.

class DistillBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistillBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

class PredictPipeline:
    def __init__(self) -> None:
        pass

    def predict(self, text):
        device = 'cuda' if cuda.is_available() else 'cpu'
        print("model class intialized")
        model = DistillBERTClass()

        model.to(device)

        # Load the saved model weights into the initialized model
        model.load_state_dict(torch.load("/opt/ml/model_state_dict_5.pth", map_location=torch.device('cpu')))
        print("model loaded with state_dict_5")

        model.eval()
        #print("model eval initiated")
        
        # Load the tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        #print('tokenizer loaded')

        #print(list(model.state_dict().items())[-3:-1])

        result = prediction(model, tokenizer, device, text)
        # result = model.predict(text)
        if result == 0:
            result = "Human"
        else:
            result = "AI"
        
        return result

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


def lambda_handler(event, context):
    try:
        text = event['text']
        prediction = PredictPipeline().predict(text)

        return {
        'statusCode': 200,
        'body': json.dumps(
            {
                "predicted_label": prediction
            }
            )
    }
    except Exception as e:
        raise