import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# --- PAGE CONFIG ---
st.set_page_config(page_title="DataLab Pro", page_icon="📊", layout="wide")

st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Session state to share data
if 'df' not in st.session_state:
    st.session_state.df = None

# --- TAB 1: AutoML ---
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
                    model = RandomForestClassifier()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    st.success(f"Model trained with accuracy: {acc:.2f}")

                    st.subheader("Confusion Matrix")
                    fig, ax = plt.subplots()
                    cm = confusion_matrix(y_test, y_pred)
                    ax.matshow(cm, cmap=plt.cm.Blues)
                    for i in range(len(cm)):
                        for j in range(len(cm[0])):
                            ax.text(j, i, cm[i, j], ha='center', va='center')
                    st.pyplot(fig)
        except Exception as e:
            st.error(f"❌ Error: {e}")
    
    st.markdown("""
    <div style='background-color: #fffde7; border-left: 6px solid #f9a825; padding: 15px; border-radius: 6px; margin-top: 20px;'>
        <h4>🚀 Build AI Projects Without Coding!</h4>
        <p>Thousands used <strong>DataLab Pro</strong> + our ₹499 Kit to create ML dashboards for internships & freelancing.</p>
        <p>💬 <em>“I used this kit to land a freelance ML project in 7 days!” – Ananya, Final Year BTech</em></p>
        <p>📽️ Watch: <a href='https://youtu.be/uq_ba4Prjps?si=KW2odA2izyFDsNw6' target='_blank'>Student Earning Proof</a> | <a href='https://indianexpress.com/article/technology/tech-layoffs-march-2025-it-layoffs-9919985/' target='_blank'>Layoff Trends (2025)</a></p>
        <a href='https://pages.razorpay.com/pl_Q9haRTHXpyB9SS/view' target='_blank' style='background:#1976d2;color:#fff;padding:10px 20px;border-radius:5px;text-decoration:none;font-weight:bold;'>💼 Get ₹499 AI Kit</a>
    </div>
    """, unsafe_allow_html=True)

# --- TAB 2: Visualization ---
def visualization_tab():
    st.header("📊 Data Visualization")
    if st.session_state.df is None:
        st.warning("Please upload a CSV in the AutoML tab first.")
        return

    df = st.session_state.df
    chart_type = st.selectbox("Choose Chart Type", ["Histogram", "Scatter", "Box Plot"])
    col1 = st.selectbox("X-Axis Column", df.columns)

    fig, ax = plt.subplots()
    if chart_type == "Histogram":
        if pd.api.types.is_numeric_dtype(df[col1]):
            df[col1].plot.hist(ax=ax)
        else:
            st.warning("Please select a numeric column.")
    elif chart_type == "Scatter":
        col2 = st.selectbox("Y-Axis Column", df.columns)
        if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
            df.plot.scatter(x=col1, y=col2, ax=ax)
        else:
            st.warning("Both columns should be numeric.")
    else:
        if pd.api.types.is_numeric_dtype(df[col1]):
            df.boxplot(column=col1, ax=ax)
        else:
            st.warning("Box plot requires numeric column.")

    st.pyplot(fig)

    st.markdown("""
    <div style='background-color: #e8f5e9; border-left: 5px solid #43a047; padding: 15px; border-radius: 6px; margin-top: 20px;'>
        <h4>📈 Land Projects with No-Code AI Dashboards</h4>
        <p>Students are using our ₹499 Kit + DataLab Pro to create AutoML visualizations and land internships & Upwork gigs.</p>
        <p>🎥 See Proof: <a href='https://youtu.be/vM8Chmkd22o?si=wIGD24ZegI8rj6Zg' target='_blank'>AI Dashboard Demo</a></p>
        <a href='https://pages.razorpay.com/pl_Q9haRTHXpyB9SS/view' target='_blank' style='background:#388e3c;color:#fff;padding:10px 20px;border-radius:5px;text-decoration:none;font-weight:bold;'>📊 Buy ₹499 Project Kit</a>
    </div>
    """, unsafe_allow_html=True)

# --- TAB 3: Data Cleaning ---
def cleaning_tab():
    st.header("🧹 Data Cleaning")
    if st.session_state.df is None:
        st.warning("Upload a CSV in the AutoML tab.")
        return

    df = st.session_state.df
    st.subheader("🔍 Missing Values")
    st.bar_chart(df.isnull().sum())

    if st.button("🗑️ Drop Rows with Missing Values"):
        st.session_state.df = df.dropna()
        st.success("Rows with missing values removed.")

    if st.button("🧮 Fill Missing with Mean"):
        df_filled = df.copy()
        for col in df.select_dtypes(include=np.number).columns:
            df_filled[col] = df_filled[col].fillna(df[col].mean())
        st.session_state.df = df_filled
        st.success("Numeric missing values filled with mean.")

    st.markdown("""
    <div style='background-color: #fce4ec; border-left: 6px solid #ec407a; padding: 15px; border-radius: 6px; margin-top: 20px;'>
        <h4>🧠 Make Your Resume AI-Ready in Minutes</h4>
        <p>Clean your datasets and use the ₹499 Kit to build AutoML-powered projects – even without Python knowledge!</p>
        <p>📰 Seen in: <a href='https://www.ndtvprofit.com/business/layoffs-2025-amazon-intel-morgan-stanley-among-companies-cutting-jobs-this-year' target='_blank'>NDTV News</a></p>
        <a href='https://pages.razorpay.com/pl_Q9haRTHXpyB9SS/view' target='_blank' style='background:#d81b60;color:#fff;padding:10px 20px;border-radius:5px;text-decoration:none;font-weight:bold;'>🎓 Buy ₹499 AI Kit</a>
    </div>
    """, unsafe_allow_html=True)

# --- MAIN UI ---
st.title("📊 DataLab Pro")
tab1, tab2, tab3 = st.tabs(["AutoML", "Visualization", "Data Cleaning"])

with tab1: automl_tab()
with tab2: visualization_tab()
with tab3: cleaning_tab()

# --- SOCIAL FOOTER ---
st.markdown("""
<hr>
<div style='text-align:center; font-size:15px; color:#555;'>
    🌟 Follow <strong>CareerUpskillers</strong> for AI career tips & tools:<br>
    <a href='https://www.linkedin.com/company/careerupskillers' target='_blank'>🔗 LinkedIn</a> • 
    <a href='https://twitter.com/careerupskill' target='_blank'>🐦 Twitter</a> • 
    <a href='https://instagram.com/careerupskillers' target='_blank'>📸 Instagram</a> • 
    <a href='https://youtube.com/@careerupskillers' target='_blank'>▶️ YouTube</a>
</div>
""", unsafe_allow_html=True)
