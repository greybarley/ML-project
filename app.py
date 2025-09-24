import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

st.set_page_config(page_title="Student Exam Performance Predictor", layout="centered")

st.title("Student Exam Performance Indicator")
st.header("Student Exam Performance Prediction")

# Create input fields
gender = st.selectbox("Gender", ["Select Gender", "male", "female"])
race_ethnicity = st.selectbox("Race or Ethnicity", ["Select Ethnicity", "group A", "group B", "group C", "group D", "group E"])
parental_level_of_education = st.selectbox(
    "Parental Level of Education",
    ["Select Parent Education", "associate's degree", "bachelor's degree", "high school", "master's degree", "some college", "some high school"]
)
lunch = st.selectbox("Lunch Type", ["Select Lunch Type", "free/reduced", "standard"])
test_preparation_course = st.selectbox("Test Preparation Course", ["Select Course", "none", "completed"])
reading_score = st.number_input("Reading Score out of 100", min_value=0, max_value=100, value=0)
writing_score = st.number_input("Writing Score out of 100", min_value=0, max_value=100, value=0)

# Predict button
if st.button("Predict Maths Score"):
    # Validate inputs
    if "Select" in [gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course]:
        st.error("Please select all dropdown fields properly!")
    else:
        # Prepare data
        data = CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=float(reading_score),
            writing_score=float(writing_score)
        )
        pred_df = data.get_data_as_data_frame()
        
        # Predict
        predict_pipeline = PredictPipeline()
        try:
            results = predict_pipeline.predict(pred_df)
            st.success(f"The predicted Maths score is: {results[0]}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
