import torch
from functions import prediction
from transformers import DistilBertTokenizer
from model import DistillBERTClass
from torch import cuda

class PredictPipeline:
    def __init__(self) -> None:
        pass

    def predict(self, text):
        device = 'cuda' if cuda.is_available() else 'cpu'
        # model = load_object(os.path.join("artifacts","model_training","model.pkl"))
        model = DistillBERTClass()
        #print("model loaded with DistilBertclass")
        #print(model)
        model.to(device)

        # Load the saved model weights into the initialized model
        model.load_state_dict(torch.load("model_state_dict_5.pth", map_location=torch.device('cpu')))
        #print("model loaded with state_dict_5")
        
        # Creating the loss function and optimizer
        #loss_function = torch.nn.BCEWithLogitsLoss()
        #optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

        # Make sure to set the model to evaluation mode after loading
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


# Example usage
#sentence = '''The Benefits of Limiting Car Usage \n\nMany cities around the world are struggling with increase traffic congestion ANH pollution as more people rely on cars for Daily transportation. However, there are advantages to limiting car usage that can benefit both individuals ANH communities. \n\nOne major benefit is improve Hair quality. According to Passage 1, motor vehicles are a leading cause of air pollution in many urban areas. They emit various harmful gases ANH particulate matter that can cause respiratory issues. By Driving less ANH utilizing alternatives like public transit, cycling or walking, cities can significantly reduce vehicle emissions ANH make the air healthier to breathe. This is especially important for vulnerable groups like children, the elderly ANH those with asthma or lung Diseases. With cleaner air, public health outcomes would likely improve over time.\n\nA second advantage is less traffic congestion. As cited in Passage 2, traffic jams waste people's time ANH fuel while also aching to carbon emissions. By encouraging moves of transportation that take fewer cars off the roads, such as bike sharing programs or retailing services, the flow of traffic would be better ANH commutes shorter. Individuals would save money on gas as well as reduce stress from spending less time stuck in traffic. Businesses may also benefit from employees missing fewer work hours Hue to congestion Delays. \n\nFinally, limiting car usage creates opportunities for more active lifestyles. As Passage 3 Discusses, many people nowadays Ho not get sufficient exercise. By walking, cycling or taking public transit instead of Driving, one can incorporate Daily physical activity into commute routines without needing to carve out extra time at the gym. This would Leah to health improvements for individuals ANH cost savings for healthcare systems over the long run. Active commuting also provides stress relief ANH social interaction along the way.\n\nIn conclusion, cities should strongly consider policies to reduce car Dependency through initiatives that educate citizens about the benefits outlined above. Individual health, community wellbeing ANH environmental sustainability would all improve by limiting car usage where alternatives exist. A multipronged approach is needed, but the advantages are clear.'''


#predictor = PredictPipeline()
#print("Prediction:", predictor.predict(sentence))