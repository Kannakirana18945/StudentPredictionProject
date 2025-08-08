import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load('random_forest_model.pkl')

# Title
st.title("🎓 Student Performance Prediction")
st.markdown("Predict whether a student will pass or fail using machine learning.")


# Input sliders
studytime = st.slider("Study Time (1–4)", 1, 4, 2)
failures = st.slider("Past Class Failures", 0, 4, 0)
absences = st.slider("Number of Absences", 0, 93, 4)
g1 = st.slider("1st Internal Marks ", 0, 20, 12)
g2 = st.slider("2nd Internal Marks ", 0, 20, 14)


# Predict button
if st.button("Predict Final Grade"):
    input_data = pd.DataFrame([[studytime, failures, absences, g1, g2]],
                              columns=['studytime', 'failures', 'absences', 'G1', 'G2'])
    prediction = model.predict(input_data)
    grade = prediction[0]

    if grade >= 10:
        st.success(f"✅ Student is likely to Pass (Predicted G3: {grade:.2f})")
    else:
        st.error(f"❌ Student is at risk of Failing (Predicted G3: {grade:.2f})")

