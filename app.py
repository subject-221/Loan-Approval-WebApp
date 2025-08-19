import streamlit as st
import pandas as pd
import joblib

# --- Page Configuration ---
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Load Model and Preprocessor ---
@st.cache_resource
def load_model_and_preprocessor():
    """Loads the saved model and preprocessor files."""
    try:
        preprocessor = joblib.load('preprocessor.joblib')
        model = joblib.load('model.joblib')
        return preprocessor, model
    except FileNotFoundError:
        return None, None

preprocessor, model = load_model_and_preprocessor()

# --- App Title and Description ---
st.title("Loan Approval Prediction 🏦")
st.markdown("This application predicts the probability of a loan getting approved based on the applicant's details. Please fill in all the fields in the sidebar to get a prediction.")
st.markdown("---")

# --- Sidebar for User Inputs ---
st.sidebar.title("👤 Applicant Details")
st.sidebar.markdown("Provide your information below.")

def get_user_inputs():
    """Creates sidebar widgets with improved layout and help tooltips."""
    st.sidebar.subheader("Personal Information")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        gender = st.sidebar.selectbox("Gender", ["Male", "Female"], help="Select your gender.")
        dependents_str = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"], help="How many people are financially dependent on you?")
    with col2:
        married = st.sidebar.selectbox("Married", ["Yes", "No"], help="Select your marital status.")
        education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"], help="Select your highest education level.")
        
    self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"], index=1, help="Are you self-employed or do you have a salaried job?")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Financial & Loan Details")
    
    col3, col4 = st.sidebar.columns(2)
    with col3:
        applicant_income = st.sidebar.number_input("Applicant Income", min_value=0, value=5000, help="Enter your total monthly income in INR.")
        loan_amount = st.sidebar.number_input("Loan Amount (in k)", min_value=10, value=150, help="Enter the desired loan amount in thousands (e.g., for 1.5 Lakhs, enter 150).")
        credit_history = st.sidebar.selectbox("Credit History", [1.0, 0.0], format_func=lambda x: "Good" if x == 1.0 else "Bad", help="Have you paid your previous debts on time?")
    with col4:
        coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0, value=1500, help="Enter your co-applicant's monthly income, if any.")
        loan_amount_term = st.sidebar.number_input("Loan Term (Months)", min_value=12, value=360, help="In how many months will you repay the loan?")
        property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"], help="Select the type of area where the property is located.")
    
    # Convert 'Dependents' from string to integer to match training data
    dependents = int(dependents_str.replace('3+', '3'))
    
    return {
        'Gender': gender, 'Married': married, 'Dependents': dependents, 
        'Education': education, 'Self_Employed': self_employed, 
        'ApplicantIncome': applicant_income, 'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount, 'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_history, 'Property_Area': property_area
    }

user_inputs = get_user_inputs()

# --- Main Page for Result Display ---
if model is None or preprocessor is None:
    st.error("⚠️ Model files are not loaded. Please ensure `model.joblib` and `preprocessor.joblib` are in the folder and the training script has been run.")
else:
    if st.sidebar.button("Predict Eligibility", type="primary", use_container_width=True):
        with st.spinner('Analyzing your profile... This may take a moment.'):
            # Create a DataFrame from user inputs
            input_data = pd.DataFrame([user_inputs])
            
            # Preprocess the data and make predictions
            input_transformed = preprocessor.transform(input_data)
            prediction = model.predict(input_transformed)
            prediction_proba = model.predict_proba(input_transformed)

            # Display the result
            st.subheader("🔮 Prediction Result")
            
            if prediction[0] == 1:
                st.success("Congratulations! Your loan application is likely to be **APPROVED**.", icon="🎉")
                st.balloons()
                
                prob_approved = prediction_proba[0][1] * 100
                st.metric(label="Approval Confidence Score", value=f"{prob_approved:.2f}%")
                st.progress(prob_approved / 100)
                
            else:
                st.error("Unfortunately, your loan application is likely to be **REJECTED**.", icon="😞")
                
                prob_rejected = prediction_proba[0][0] * 100
                st.metric(label="Rejection Confidence Score", value=f"{prob_rejected:.2f}%")
                st.progress(prob_rejected / 100)

            # Show the user's provided details in an expander
            with st.expander("Show the details you provided"):
                st.dataframe(input_data)