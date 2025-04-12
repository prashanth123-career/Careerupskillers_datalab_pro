import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# --- Main App Configuration ---
st.set_page_config(page_title="DataLab Pro", page_icon="📊", layout="wide")

# ---------- HIDE STREAMLIT DEFAULT HEADER & FOOTER ----------
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        html, body, [class*="css"]  {
            font-family: 'Segoe UI', sans-serif;
            font-size: 16px;
            color: #333333;
        }

        .ad-box {
            background-color: #fffde7;
            border-left: 6px solid #f9a825;
            padding: 20px;
            border-radius: 10px;
            margin: 25px 0 35px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        }

        .ad-box h4 {
            margin-top: 0;
            color: #f57f17;
        }

        .ad-button {
            display: inline-block;
            background-color: #1976d2;
            color: white;
            padding: 10px 25px;
            border-radius: 6px;
            text-decoration: none;
            font-weight: bold;
            margin-top: 10px;
        }

        .banner-title {
            font-size: 36px;
            font-weight: bold;
            color: #2e3b4e;
            text-align: center;
            margin-top: 20px;
            margin-bottom: 10px;
        }

        .banner-subtitle {
            text-align: center;
            color: #607d8b;
            font-size: 18px;
            margin-bottom: 30px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='banner-title'>📊 DataLab Pro</div>
<div class='banner-subtitle'>Your No-Code ML, Visualization & Cleaning Suite</div>
""", unsafe_allow_html=True)

if 'df' not in st.session_state:
    st.session_state.df = None

def automl_tab():
    st.header("🔮 AutoML")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.write("Data Preview:")
            st.dataframe(df.head())
            target = st.selectbox("Select Target Column", df.columns)
            if st.button("Train Model"):
                with st.spinner("Training model..."):
                    X = pd.get_dummies(df.drop(target, axis=1))
                    y = df[target]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = RandomForestClassifier(random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    st.success(f"Model trained! Accuracy: {acc:.2f}")
                    st.subheader("Confusion Matrix")
                    fig, ax = plt.subplots()
                    cm = confusion_matrix(y_test, y_pred)
                    ax.matshow(cm, cmap=plt.cm.Blues)
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            ax.text(j, i, cm[i, j], ha='center', va='center')
                    st.pyplot(fig)
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

    # ✅ ₹499 Kit Ad Section
    st.markdown("""
    <div class='ad-box'>
      <h4>🚀 Used By Students & Freelancers to Land Projects</h4>
      <p>Arjun (lost job in Nov 2024) now earns ₹90K–1.7L/month freelancing ML projects.</p>
      <p>🎓 Our ₹499 AI Kit helps you:</p>
      <ul>
        <li>Train AutoML dashboards for clients</li>
        <li>Crack freelance & data science interviews</li>
        <li>Build CSV → ML projects instantly</li>
      </ul>
      <p>🎥 <a href='https://youtu.be/uq_ba4Prjps?si=KW2odA2izyFDsNw6' target='_blank'>Proof: ₹1L+/mo earnings</a> • 
         <a href='https://indianexpress.com/article/technology/tech-layoffs-march-2025-it-layoffs-9919985/' target='_blank'>Layoffs 2025: The Urgency</a></p>
      <a href='https://pages.razorpay.com/pl_Q9haRTHXpyB9SS/view' target='_blank' class='ad-button'>🔥 Get ₹499 AI Kit Now</a>
    </div>
    """, unsafe_allow_html=True)

st.title("")
tab1 = st.tabs(["AutoML"])[0]
with tab1:
    automl_tab()
