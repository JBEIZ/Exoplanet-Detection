import streamlit as st
import pandas as pd
import pickle
import urllib.request

# Load the trained model from GitHub
MODEL_URL = "https://github.com/JBEIZ/Exoplanet-Detection/edit/main/exoplanet_detection.pkl"  # Update with your actual GitHub link
urllib.request.urlretrieve(MODEL_URL, "exoplanet_detection.pkl")

with open("exoplanet_detection.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit UI
st.set_page_config(page_title="🔭 Exoplanet Detection AI", layout="wide")

# Title and Description
st.title("🔭 Exoplanet Detection AI")
st.markdown("### Upload telescope data to predict exoplanets!")
st.write("This tool uses a **Machine Learning model** to detect potential exoplanets from telescope observations.")

# Upload CSV File
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded file
    data = pd.read_csv(uploaded_file)

    # Required features
    required_features = ["koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_insol"]
    
    if not all(col in data.columns for col in required_features):
        st.error("❌ The uploaded file is missing required columns!")
    else:
        # Make predictions
        predictions = model.predict(data[required_features])
        data["Exoplanet Prediction"] = ["🪐 Exoplanet" if p == 1 else "🚫 No Exoplanet" for p in predictions]

        # Show results
        st.write("### 📊 Prediction Results")
        st.dataframe(data)

        # Download button for results
        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions", csv, "exoplanet_predictions.csv", "text/csv")

# Footer
st.markdown("---")
st.write("🚀 **Developed by [Your Name]** | Powered by **Machine Learning & Streamlit**")
