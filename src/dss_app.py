import streamlit as st
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier


data = './data/user_behavior_dataset.csv'

df = pd.read_csv(data)
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])

X = df[['Gender', 'Age', 'Screen On Time (hours/day)', 'Number of Apps Installed']]
y = df['Battery Drain (mAh/day)']

pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))
pipeline.fit(X,y)

st.title('Battery Drain Prediction Based on User Behavior')

user_gender = st.selectbox('Gender',['Male', 'Female'])
user_age = st.number_input('Age (1-100)', min_value=1,max_value=100,value=30, step=1)
user_screen_on_time = st.slider("Screen On Time hours/day (0-24)", min_value=0.0, max_value=24.0, value=5.0, step=0.1)
user_num_apps = st.number_input("Number of Apps Installed (1-500)", min_value=1, max_value=500, value=50, step=1)


def predict_behavior_class(gender, age, screen_on_time, num_apps_installed):
    gender_encoded = label_encoder.transform([gender])[0]
    user_input = pd.DataFrame([[gender_encoded,age,screen_on_time,num_apps_installed]],
                              columns=['Gender', 'Age', 'Screen On Time (hours/day)', 'Number of Apps Installed'])
    
    predicted_battery_drain = pipeline.predict(user_input)
    return predicted_battery_drain[0]

if st.button('Predict Battery Drain (mAh/day)'):
    predicted_drain = predict_behavior_class(user_gender, user_age, user_screen_on_time,user_num_apps)
    st.write(f'Predicted Battery Drain: {predicted_drain:.2f} mAh/day')