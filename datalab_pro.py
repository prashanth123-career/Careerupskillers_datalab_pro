import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Ensure compatibility with Streamlit
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# --- Main App Configuration ---
st.set_page_config(
    page_title="DataLab Pro",
    page_icon="📊",
    layout="wide"
)

# ---------- HIDE STREAMLIT DEFAULT HEADER & FOOTER ----------
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None

# --- Tab 1: AutoML ---
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
                    # Convert categorical data to numeric
                    X = df.drop(target, axis=1)
                    y = df[target]
                    
                    # Simple encoding for categorical data
                    X = pd.get_dummies(X)
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = RandomForestClassifier(random_state=42)
                    model.fit(X_train, y_train)
                    
                    # Results
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    st.success(f"Model trained! Accuracy: {acc:.2f}")
                    
                    # Confusion matrix
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

# --- Tab 2: Visualization ---
def visualization_tab():
    st.header("📊 Data Visualization")
    
    if st.session_state.df is None:
        st.warning("Please upload data in the AutoML tab first!")
        return
    
    df = st.session_state.df
    chart_type = st.selectbox("Select Chart Type", 
                            ["Histogram", "Scatter Plot", "Box Plot"])
    
    col1 = st.selectbox("Select Column", df.columns)
    
    fig, ax = plt.subplots()
    try:
        if chart_type == "Histogram":
            if pd.api.types.is_numeric_dtype(df[col1]):
                df[col1].hist(ax=ax)
            else:
                st.error("Please select a numeric column for Histogram.")
                return
        elif chart_type == "Scatter Plot":
            col2 = st.selectbox("Select Y-Axis Column", df.columns)
            if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                df.plot.scatter(x=col1, y=col2, ax=ax)
            else:
                st.error("Please select numeric columns for Scatter Plot.")
                return
        else:  # Box Plot
            if pd.api.types.is_numeric_dtype(df[col1]):
                df.boxplot(column=col1, ax=ax)
            else:
                st.error("Please select a numeric column for Box Plot.")
                return
        
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating plot: {str(e)}")
    
    # Advertisement for Tab 2
    st.markdown("""
    <div style='background-color: #fffbe6; border-left: 5px solid #ff9800; padding: 15px; border-radius: 8px; margin-top: 20px;'>
      <h4>🛡️ Become a Freelance VAPT Expert</h4>
      <p>Many students used our ₹499 Kit to start offering web security & vulnerability scans on Fiverr & Internshala. Clients now pay them ₹2K–₹8K per scan!</p>
      <p>🎥 Watch:
        <a href='https://youtu.be/vM8Chmkd22o?si=wIGD24ZegI8rj6Zg' target='_blank'>Freelance Security Path</a> •
        <a href='https://youtu.be/uq_ba4Prjps?si=KW2odA2izyFDsNw6' target='_blank'>₹1L Freelancing Proof</a>
      </p>
      <a href='https://pages.razorpay.com/pl_Q9haRTHXpyB9SS/view' target='_blank' style='background-color:#1976d2;color:white;padding:10px 20px;border-radius:5px;text-decoration:none;'>💻 Buy Kit & Start Scanning</a>
    </div>
    """, unsafe_allow_html=True)

# --- Tab 3: Data Cleaning ---
def cleaning_tab():
    st.header("🧹 Data Cleaning")
    
    if st.session_state.df is None:
        st.warning("Please upload data in the AutoML tab first!")
        return
    
    df = st.session_state.df
    
    # Missing values analysis
    st.subheader("Missing Values Summary")
    missing_data = df.isnull().sum()
    st.bar_chart(missing_data)
    
    # Cleaning options
    st.subheader("Cleaning Tools")
    if st.button("Remove Rows with Missing Values"):
        cleaned_df = df.dropna()
        st.session_state.df = cleaned_df
        st.success(f"Removed {len(df) - len(cleaned_df)} rows")
    
    if st.button("Fill Missing Values with Mean"):
        try:
            cleaned_df = df.copy()
            for col in cleaned_df.select_dtypes(include=np.number).columns:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            st.session_state.df = cleaned_df
            st.success("Filled missing numeric values with column means")
        except Exception as e:
            st.error(f"Error filling missing values: {str(e)}")
    
    # Advertisement for Tab 3
    st.markdown("""
    <div style='background-color: #e0f7fa; border-left: 5px solid #00bcd4; padding: 15px; border-radius: 8px; margin-top: 20px;'>
      <h4>📊 Master SIEM Monitoring – Even As A Fresher!</h4>
      <p>This ₹499 kit includes a ready-to-use SIEM dashboard + demo alerts. Great for practicing incident detection, threat correlation, and impressing recruiters.</p>
      <p>🗞 Featured in:
        <a href='https://www.ndtvprofit.com/business/layoffs-2025-amazon-intel-morgan-stanley-among-companies-cutting-jobs-this-year' target='_blank'>NDTV</a> •
        <a href='https://youtu.be/3ZmtSdAjxCM?si=h7W4AaezK_6xaBQd' target='_blank'>SIEM Career Insights</a>
      </p>
      <a href='https://pages.razorpay.com/pl_Q9haRTHXpyB9SS/view' target='_blank' style='background-color:#00838f;color:white;padding:10px 20px;border-radius:5px;text-decoration:none;'>🎯 Get Your SIEM Toolkit</a>
    </div>
    """, unsafe_allow_html=True)

# --- Main App ---
st.title("📊 DataLab Pro")
tab1, tab2, tab3 = st.tabs(["AutoML", "Visualization", "Data Cleaning"])

with tab1:
    automl_tab()

with tab2:
    visualization_tab()

with tab3:
    cleaning_tab()

# --- Social Media Footer (All Tabs) ---
st.markdown("""
<hr style='margin-top:40px;'>
<div style='text-align:center; font-size:16px; color:gray;'>
    🧭 Follow <strong>CareerUpskillers</strong> for Career Tips & AI Tools:<br><br>
    <a href='https://www.linkedin.com/company/careerupskillers' target='_blank' style='margin: 0 10px;'>🔗 LinkedIn</a> • 
    <a href='https://twitter.com/careerupskill' target='_blank' style='margin: 0 10px;'>🐦 Twitter</a> • 
    <a href='https://instagram.com/careerupskillers' target='_blank' style='margin: 0 10px;'>📸 Instagram</a> • 
    <a href='https://youtube.com/@careerupskillers' target='_blank' style='margin: 0 10px;'>▶️ YouTube</a>
</div>
""", unsafe_allow_html=True)
