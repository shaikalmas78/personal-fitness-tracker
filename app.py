import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('personal_fitness_tracker.pkl', 'rb'))

# Load the dataset
df = pd.read_csv('fitness.csv')

# App Title
st.title("Personal Fitness Tracker")

st.markdown("""
In this WebApp you can observe your predicted calories burned in your body. Pass your parameters such as **Age**, **Gender**, **BMI**, etc., and see the predicted value of kilocalories burned.
""")

# Sidebar for user input
st.sidebar.header('User Input Parameters')

def user_input_features():
    age = st.sidebar.slider('Age', 10, 100, 30)
    bmi = st.sidebar.slider('BMI', 15, 40, 24)
    duration = st.sidebar.slider('Duration (min)', 0, 35, 15)
    heart_rate = st.sidebar.slider('Heart Rate', 60, 130, 80)
    body_temp = st.sidebar.slider('Body Temperature (°C)', 36, 42, 38)
    gender = st.sidebar.radio('Gender', ('Male', 'Female'))

    # Calculate height and weight from BMI (optional or replace as per your data)
    height = 170  # Example static value (in cm)
    weight = (bmi * (height / 100) ** 2)

    gender_male = 1 if gender == 'Male' else 0

    data = {
        'Age': age,
        'Height': height,
        'Weight': weight,
        'Duration': duration,
        'Heart_Rate': heart_rate,
        'Body_Temp': body_temp,
        'Gender_male': gender_male
    }

    features = pd.DataFrame(data, index=[0])
    return features, age, duration, heart_rate, body_temp

input_df, user_age, user_duration, user_heart_rate, user_body_temp = user_input_features()

# Main Panel
st.subheader('Your Parameters:')
st.write(input_df)

# Prediction
prediction = model.predict(input_df)[0]
st.subheader('Prediction:')
st.success(f'You burned approximately {prediction:.2f} kilocalories!')

# Show Similar Results
similar_results = df[
    (df['Age'].between(user_age - 5, user_age + 5)) &
    (df['Heart_Rate'].between(user_heart_rate - 5, user_heart_rate + 5))
]

st.subheader('Similar Results:')
st.dataframe(similar_results.head(5))

# General Information Percentiles
st.subheader('General Information:')

def calculate_percentile(value, series):
    """Calculate the percentile of a value compared to a pandas Series."""
    return np.round((series < value).mean() * 100, 1)

# Percentiles for General Information
age_percentile = calculate_percentile(user_age, df['Age'])
duration_percentile = calculate_percentile(user_duration, df['Duration'])
heart_rate_percentile = calculate_percentile(user_heart_rate, df['Heart_Rate'])
body_temp_percentile = calculate_percentile(user_body_temp, df['Body_Temp'])

# Display General Information
st.markdown(f"""
- You are older than **{age_percentile}%** of other people.
- Your exercise duration is higher than **{duration_percentile}%** of other people.
- You have a higher heart rate than **{heart_rate_percentile}%** of other people during exercise.
- You have a higher body temperature than **{body_temp_percentile}%** of other people during exercise.
""")

# Footer (Optional)
sst.markdown("---")
#st.caption("Made with ❤️ using Streamlit")
