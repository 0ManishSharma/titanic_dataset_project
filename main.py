import streamlit as st
import requests

st.title("🚢 Titanic Predictor")
st.markdown("**FastAPI + ML Model**")

# Sidebar inputs
pclass = st.sidebar.selectbox("Pclass", [1,2,3])
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 0, 80, 25)
fare = st.sidebar.slider("Fare", 0.0, 500.0, 25.0)
sibsp = st.sidebar.slider("SibSp", 0, 8, 0)
parch = st.sidebar.slider("Parch", 0, 6, 0)
embarked = st.sidebar.selectbox("Embarked", ["S", "C", "Q"])

if st.sidebar.button("🔮 Predict", type="primary"):
    features = {
        'Pclass': pclass,
        'Sex': sex,
        'Age': float(age),
        'Fare': float(fare),
        'SibSp': sibsp,
        'Parch': parch,
        'Embarked': embarked
    }
    
    try:
        response = requests.post("http://localhost:8000/predict", json=features)
        result = response.json()
        
        st.success("✅ Prediction!")
        col1, col2,col3 = st.columns(3)
        col1.metric("Survived", "✅ YES" if result['Survived'] == 1 else "❌ NO")
        col2.metric("Survival %", f"{result['Survival_Probability']:.0%}")
        col3.metric("Death %", f"{result['Death_Probability']:.0%}")
        
    except:
        st.error("❌ Start FastAPI: `uvicorn api:app --port 8000`")

st.markdown("[API Docs](http://localhost:8000/docs)")
