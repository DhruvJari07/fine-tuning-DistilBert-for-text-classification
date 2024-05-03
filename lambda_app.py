import requests
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

Lambda_api_url = os.getenv("LAMBDA_API_URL")

def predict_class_aws(text: str):
    API_URL = Lambda_api_url

    req = {
            "text": text
        }

    r = requests.post(API_URL, json=req)

    return r.json()['predicted_label']

def main():
    # Set title and description
    st.title("AI & Human Text Classification")
    st.write("To detect if your text is written by human or AI, enter your text below.")

    # Text input
    user_input = st.text_area("Enter your input text here:", "", height=200)

    # Model selection
    #selected_model = st.selectbox("Select Model", ["Bert-tuned Model", "ML Model"])


    # Predict button
    if st.button("Predict"):
        with st.spinner("Predicting..."):
            #if selected_model == "ML Model":
            #    pass
                # Instantiate PredictPipeline
        #        predictor = PredictPipeline()
            #elif selected_model == "Bert-tuned Model":
                # Instantiate PredictPipeline2
            prediction = predict_class_aws(user_input)

            # Display prediction result
            st.write(f"Prediction: Your text is **{prediction}** generated.")


if __name__ == "__main__":
    main()