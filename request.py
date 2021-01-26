import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Pregnancies':7, 'Glucose':115, 'BloodPressure':80, 'SkinThickness':30, 'Insulin':85, 'BMI':30.5, 'DiabetesPedigreeFunction':0.704, 'Age':40})

print(r.json())
