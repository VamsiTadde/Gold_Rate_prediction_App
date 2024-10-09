import streamlit as st
import pickle
import numpy as np
import sklearn


# Load the saved model
import pickle

# with open('gold_rate_pred_updated.pkl', 'rb') as file:
    # model = pickle.load(file)
    
model = pickle.load(open('./gold_rate_pred_updated.pkl', 'rb'))
print(model)
# Set the title of the Streamlit app
st.title("Gold Rate Prediction App")

# Add a brief description
st.write("This app predicts the gold rate based on the year using a simple linear regression model.")

# Add input widget for user to enter years of experience
year_of_goldrate = st.number_input("Enter Year of Gold Rate You want:", min_value=0, max_value=2050)

# When the button is clicked, make predictions
if st.button("Predict Gold Rate"):
    # Make a prediction using the trained model
    year_input = np.array([[year_of_goldrate]])  # Convert the input to a 2D array for prediction
    prediction = model.predict(year_input)
   
    # Display the result
    st.success(f"The predicted salary for {year_of_goldrate} year of gold rate is: {prediction[0]:,.2f}")
   
# Display information about the model
st.write("The model was trained using a dataset of goldartes and years .")
