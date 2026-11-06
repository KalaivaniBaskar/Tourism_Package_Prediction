import os
import joblib
import pandas as pd
import streamlit as st

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Prediction App")
st.write("The Tourism Package Prediction App is an internal tool that predicts whether a customer will purchase the newly introduced Wellness Tourism Package.")
st.write("Kindly enter the customer details to check whether they are likely to buy the tourism package.")

with st.form("customer_input_form"):
    st.subheader("Customer Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
        city_tier = st.selectbox("City Tier", ["tier1", "tier2", "tier3"])
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        no_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
        preferred_property_star = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5], index=0)
        marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])

    with col2:
        no_of_trips = st.number_input("Number of Trips", min_value=1, max_value=400, value=2)
        passport = st.selectbox("Has Passport?", ["yes", "no"])
        own_car = st.selectbox("Owns a Car?", ["yes", "no"])
        no_of_children_visiting = st.number_input("Number of Children Visiting (< age 5)", min_value=0, max_value=15, value=0)
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        monthly_income = st.number_input("Monthly Income", min_value=0.0, value=20000.0)

    st.subheader("Customer Interaction Data")
    
    col3, col4 = st.columns(2)
    
    with col3:
        pitch_satisfaction_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5], index=2)
        product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])

    with col4:
        no_of_followups = st.number_input("Number of Follow-ups", min_value=1, max_value=10, value=3)
        duration_of_pitch = st.number_input("Duration of Pitch (Minutes)", min_value=0, value=15)

    # Submit button inside the form
    submit_button = st.form_submit_button("Predict")

# Run prediction upon form submission
if submit_button:
    # Convert inputs to match model training DataFrame structure
    input_data = pd.DataFrame([{
        "Age": age,
        "TypeofContact": type_of_contact,
        "CityTier": city_tier,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": no_of_person_visiting,
        "PreferredPropertyStar": preferred_property_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": no_of_trips,
        "Passport": passport,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": no_of_children_visiting,
        "Designation": designation,
        "MonthlyIncome": monthly_income,
        "PitchSatisfactionScore": pitch_satisfaction_score,
        "ProductPitched": product_pitched,
        "NumberOfFollowups": no_of_followups,
        "DurationOfPitch": duration_of_pitch
    }])

    # Set the classification threshold
    classification_threshold = 0.45

    # Predict button evaluation
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    
    result = "likely" if prediction == 1 else "unlikely"
    st.write(f"Based on the information provided, the customer is **{result} to purchase** the tourism package (Probability: {prediction_proba:.2%}).")
