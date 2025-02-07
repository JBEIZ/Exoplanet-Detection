import streamlit as st
import pandas as pd
import pickle

# Load the trained model from the same folder
with open("exoplanet_detection.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🔭 Exoplanet Detection AI")
st.write("Upload telescope data to detect potential exoplanets!")

# File uploader
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    required_features = ["koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_insol"]
    
    if not all(col in data.columns for col in required_features):
        st.error("❌ The uploaded file is missing required columns!")
    else:
        predictions = model.predict(data[required_features])
        data["Exoplanet Prediction"] = ["🪐 Exoplanet" if p == 1 else "🚫 No Exoplanet" for p in predictions]

        st.write("### 📊 Prediction Results")
        st.dataframe(data)

        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions", csv, "exoplanet_predictions.csv", "text/csv")

