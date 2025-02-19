import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the pre-trained model
try:
    # Load the model correctly in binary mode
    with open('lr.pkl', 'rb') as file:
        model = joblib.load(file)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Set the title of the app
st.title("Diabetes Detection App")

st.image(
    "/workspaces/Diabetes-Detection/diabe.png",  # Path to your image
    caption="Early detection of diabetes is crucial for better health management.",
    use_container_width=True
)

# Sidebar options for input
st.sidebar.header("Input Options")
option = st.sidebar.radio("Choose input method:", ("Manual Entry", "Upload CSV"))

# Function to make predictions
def predict_diabetes(data):
    prediction = model.predict(data)
    probability = model.predict_proba(data)[0][1]  # Probability of having diabetes
    return prediction[0], probability

# Manual data entry
if option == "Manual Entry":
    st.subheader("Manual Data Entry")

    # Create columns for horizontal layout
    col1, col2, col3, col4 = st.columns(4)

    # Inputs for user data arranged horizontally
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, step=1)
    with col2:
        glucose = st.number_input("Glucose Level", min_value=0, max_value=300, step=1)
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, step=1)
    with col3:
        insulin = st.number_input("Insulin Level", min_value=0, max_value=900, step=1)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, step=0.1)
    with col4:
        diabetic_pedigree = st.number_input(
            "Diabetic Pedigree Function", 
            min_value=0.0, max_value=2.5, value=0.5, step=0.01
        )
        age = st.number_input("Age", min_value=16, max_value=120, step=1)

    # Prepare data for prediction
    input_data = np.array([[pregnancies, glucose, blood_pressure, 
                            skin_thickness, insulin, bmi, diabetic_pedigree, age]])


# Prediction button
if st.button("Predict"):
    try:
        prediction, probability = predict_diabetes(input_data)
        probability_percentage = probability * 100
        non_diabetic_percentage = (1 - probability) * 100

        if prediction == 1:
            st.success(f"There is a {probability_percentage:.2f}% chance that you may have diabetes.")
        else:
            st.success(f"There is a {non_diabetic_percentage:.2f}% chance that you may not have diabetes.")
    except Exception as e:
        st.error(f"Error during prediction: {e}")


# File upload for batch predictions
elif option == "Upload CSV":
    st.sidebar.subheader("Upload a CSV File")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # Read the uploaded CSV file
            df = pd.read_csv(uploaded_file)

            # Required columns for prediction
            expected_columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
                                "Insulin", "BMI", "DiabeticPedigreeFunction", "Age"]

            if all(column in df.columns for column in expected_columns):
                st.write("## Uploaded CSV Data:")
                st.dataframe(df)

                # Prediction button for CSV data
                if st.button("Predict for CSV Data"):
                    predictions = model.predict(df)
                    probabilities = model.predict_proba(df)[:, 1]  # Probabilities of having diabetes

                    # Add predictions to the DataFrame
                    df['Prediction'] = predictions
                    df['Probability'] = probabilities
                    st.write("## Prediction Results:")
                    st.dataframe(df)
                    
                    # Provide an option to download the results
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Predictions as CSV", csv, "predictions.csv", "text/csv")
            else:
                st.error("The uploaded CSV does not contain the required columns.")
        except Exception as e:
            st.error(f"Error reading or processing the CSV file: {e}")
    else:
        st.info("Please upload a CSV file to display the data.") 