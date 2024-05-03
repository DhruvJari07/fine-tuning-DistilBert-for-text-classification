# For local inference of model using streamlit


import streamlit as st
from predict import PredictPipeline

def main():
    # Set title and description
    st.title("AI & Human Text Classification")
    st.write("To detect if your text is written by human or AI, enter your text below.")

    # Text input
    user_input = st.text_area("Enter your input text here:", "")

    # Model selection
    # selected_model = st.selectbox("Select Model", ["ML Model", "CNN Model"])


    # Predict button
    if st.button("Predict"):
    #    if selected_model == "Default Model":
            # Instantiate PredictPipeline
    #        predictor = PredictPipeline()
    #    elif selected_model == "CNN Model":
            # Instantiate PredictPipeline2
    #       predictor = PredictPipeline2()"""
        
        # Instantiate PredictPipeline
        predictor = PredictPipeline()

        # Make prediction
        prediction = predictor.predict(user_input)

        # Display prediction result
        st.write(f"Prediction: Your text is **{prediction}** generated.")


if __name__ == "__main__":
    main()
