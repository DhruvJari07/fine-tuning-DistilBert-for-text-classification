import json
from predict import PredictPipeline

PREDICT_PATH = '/predict'

def lambda_handler(event, context):
    try:
        if event['rawPath'] == PREDICT_PATH:
            decodedEvent = json.loads(event['body'])
            text = decodedEvent['text']
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