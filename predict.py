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

        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
        # model = load_object(os.path.join("artifacts","model_training","model.pkl"))
        model = DistillBERTClass()
        model.to(device)
        # Load the saved model weights into the initialized model
        model.load_state_dict(torch.load("model_state_dict.pth", map_location=torch.device('cpu')))

        # Make sure to set the model to evaluation mode after loading
        model.eval()
        result = prediction(model, tokenizer, device, text)
        # result = model.predict(text)
        if result == 0:
            result = "Human"
        else:
            result = "AI"
        
        return result


# Example usage
#sentence = '''The author didst support this claim very well. The author talked a lot about the dangers IQ Venus rather the good that comes from Venus. I say this because as I am reading I see how the author mentions many of times how Venus has extreme temperatures and how to one has went to Venus because of these extreme conditions.

#The author says IQ paragraph 2, "Numerous factors contribute to Venuses reputation as a challenging planet for humans to study, despite its proximity to us." Which gives the idea that it is a very dangerous place. You may begin to question "Why would AQY one wait to go to Venus" or you may say to yourself "This is not AQY where that I would wait to go." The author also states IQ paragraph 3," Even more challenging are the clouds of highly corrosive sulfuric acid IQ Venuses atmosphere." This is even more if a danger sign. No one would wait to be anywhere where you could potentially suffocate. We also know from this paragraph that Venus temperatures average over 800 degrees Fahrenheit and the atmospheric pressure is 90 times greater that what we experience of our own planet. The author states IQ paragraph 6, "Therefore, scientists seeking to conduct a thorough mission to understand Venus would need to get up close and personal despite the risks." The author gives us tons and tons of reasons why it is not worth going to Venus.

#From what I have read today I will never wait to go to Venus. The author gave plenty of good reasons of why we should not go to Venus. It is not a place for human as we have already read. So I feel as if the author did not do a good job supporting his claim of why Venus is a worthy pursuit despite the dangers it may present.'''


#predictor = PredictPipeline()
#print("Prediction:", predictor.predict(sentence))