import keras as keras
import numpy as np

print("This is a neural network which predicts the presence of heart disease(0-4)")
print("using a processed dataset from the Hungarian Institute of Cardiology.")
print("Enter the following information about the patient to see the model's prediction.")

age = float(input("Age(Integer): "))
sex = float(input("Sex(1: Male, 0: Female): "))
cp = float(input("Chest Pain type(1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic): "))
trestbps = float(input("Resting blood pressure(mm Hg): "))
chol = float(input("Serum Cholesterol(mg/dl): "))
fbs = float(input("fasting blood sugar > 120 mg/dl (1: true, 0: false): "))
restecg = float(input("resting electrocardiographic results(0: normal, 1: having ST-T wave abnormality, 2: probable or definite left ventricular hypertrophy): "))
thalach = float(input("maximum heart rate achieved(bpm): "))
exang = float(input("exercise induced angina (1: yes; 0: no): "))
oldpeak = float(input("ST depression induced by exercise relative to rest: "))

model = keras.models.load_model('heart_model')

input = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak]])

results = model.predict(input)
for result in results:
    for _result in result:
        print("*******************")
        print("predicted value: " + str(round(_result)) + " (0: no presence, 4: very likely)")
        print("*******************")