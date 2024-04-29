from predict import PredictPipeline


def lambda_handler(event, context):
    try:
        user_text = event['text']
        predictor = PredictPipeline()

        # Make prediction
        prediction = predictor.predict(user_text)

        return prediction
    except Exception as e:
        raise