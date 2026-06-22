import streamlit as st
import joblib
import numpy as np
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="House Price Prediction", page_icon="🏡", layout="centered")

st.title("🏡 Advanced House Price Prediction System")
st.markdown("Enter the structural and geographic features of the house to predict its market price.")

# --- 2. LOAD THE BEST MODEL ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../Model/xgboost.joblib')

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except FileNotFoundError:
    st.error(f"Model file not found at {MODEL_PATH}. Please make sure Phase 6 was completed.")
    st.stop()

# --- 3. UI INPUTS (Matching exactly with your UI Requirements) ---
st.header("Property Features")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("1. Area (sq ft)", min_value=500, max_value=10000, value=2000, step=100)
    bedrooms = st.number_input("2. No. of Bedrooms", min_value=1, max_value=10, value=3, step=1)
    bathrooms = st.number_input("3. No. of Bathrooms", min_value=1, max_value=10, value=2, step=1)
    floors = st.number_input("4. Total Floors", min_value=1, max_value=5, value=1, step=1)

with col2:
    year_built = st.number_input("5. Year Built", min_value=1800, max_value=2026, value=2000, step=1)
    
    # 6. Location selection
    location_options = {'Rural': 0, 'Suburban': 1, 'Urban': 2, 'Downtown': 3}
    location_selection = st.selectbox("6. Location", options=list(location_options.keys()))
    
    # 7. Condition selection (1 to 5 scale)
    condition = st.slider("7. Property Condition (1 to 5 scale)", min_value=1, max_value=5, value=3, step=1)
    
    # 8. Garage selection
    garage_selection = st.selectbox("8. Garage Available", options=['No', 'Yes'])

# --- 4. DYNAMIC FEATURE ENGINEERING ---

CURRENT_YEAR = 2026
property_age = CURRENT_YEAR - year_built

# Map categorical inputs to their respective encoded numbers
location_encoded = location_options[location_selection]
garage_encoded = 1 if garage_selection == 'Yes' else 0

# --- 5. PREDICTION LOGIC ---
st.markdown("---")

if st.button("Predict House Price 🔮", use_container_width=True):
    input_features = np.array([[
        area, 
        bedrooms, 
        bathrooms, 
        floors, 
        location_encoded, 
        condition, 
        garage_encoded, 
        property_age
    ]])
    
    # Execute the prediction
    predicted_price = model.predict(input_features)[0]
    
    # Display the result
    st.success(f"### Estimated Property Value: **${predicted_price:,.2f}**")
    st.balloons()
