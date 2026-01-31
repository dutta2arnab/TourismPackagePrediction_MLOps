import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="dutta2arnab/tourism-package-prediction", filename="best_toursim_prediction_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Tourism Package Prediction App")
st.write("The ourism Package Prediction App is an internal tool for  that predicts whether customers are at risk of churning based on their details.")
st.write("Kindly enter the customer details to check whether they are likely to churn.")

# Collect user input
Age = st.number_input("Age (customer's Age)", min_value=0, max_value=100, value=30)
TypeofContact = st.selectbox("TypeofContact (Contact type)", ["Company Invited", "Self Inquiry"])
CityTier = st.selectbox("CityTier (city category)", ["1", "2", "3"])
Occupation = st.selectbox("Occupation (customer's occupation)", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
Gender = st.selectbox("Gender (customer's gender)", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting (number of persons the customer is visiting)", min_value=0, value=9)
PreferredPropertyStar = st.number_input("Preferred Property Star (customer's preferred property star rating)", min_value=1.0, max_value=5.0,value=3.0)
MaritalStatus = st.selectbox("Marital Status (customer's marital status)", ["Married", "Divorced", "Single"])
NumberOfTrips = st.number_input("Number of Trips (number of trips the customer has made)", min_value=0)
Passport = st.selectbox("Passport (customer's passport status)", ["Yes", "No"])
OwnCar = st.selectbox("Own Car?", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting (number of children the customer is visiting)", min_value=0)
Designation = st.selectbox("Designation (customer's designation)", ["Executive", "Managerial", "Professional", "Other"])
MonthlyIncome = st.number_input("Monthly Income (customer's monthly income)")
ProductPitched = st.selectbox("Product Pitched (product pitched by the customer)", ["Deluxe", "Standard", "Basic"])
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5,value=3)
NumberOfFollowups = st.number_input("Number of Followups (number of follow-ups the customer has made)", min_value=0)
DurationOfPitch = st.number_input("Duration of Pitch (duration of the pitch in minutes)", min_value=0)


# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age': Age,
    'CityTier': CityTier,
    'DurationOfPitch' : DurationOfPitch,
    'NumberOfPersonVisiting' : NumberOfPersonVisiting,
    'NumberOfFollowups' : NumberOfFollowups,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome,
    'TypeofContact': TypeofContact,    
    'Occupation': Occupation,
    'Gender': Gender,    
    'ProductPitched' : ProductPitched,    
    'MaritalStatus': MaritalStatus,    
    'Designation': Designation    
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "purchase package" if prediction == 1 else "not purchase package"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
