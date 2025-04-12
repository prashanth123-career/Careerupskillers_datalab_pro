import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Page Config
st.set_page_config(
    page_title="DataLab Light",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None

# --- Tab 1: AutoML ---
def automl_tab():
    st.header("🔮 AutoML (Basic)")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success(f"Data loaded! Shape: {df.shape}")
        
        # Target selection
        target = st.selectbox("Select Target Column", df.columns)
        
        if st.button("Train Model"):
            with st.spinner("Training..."):
                # Simple Random Forest
                X = df.drop(target, axis=1)
                y = df[target]
                
                # Handle categorical data (simple version)
                X = pd.get_dummies(X)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Results
                st.success(f"Model trained! Accuracy: {accuracy_score(y_test, y_pred):.2f}")
                
                # Confusion matrix plot
                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots()
                cm = confusion_matrix(y_test, y_pred)
                ax.matshow(cm, cmap=plt.cm.Blues)
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, cm[i, j], ha='center', va='center')
                st.pyplot(fig)

# --- Tab 2: Data Visualization ---
def visualization_tab():
    st.header("📈 Basic Visualizations")
    
    if st.session_state.df is None:
        st.warning("Upload data in the AutoML tab first!")
        return
    
    df = st.session_state.df
    
    # Chart type selection
    chart_type = st.selectbox(
        "Choose Chart Type",
        ["Scatter Plot", "Histogram", "Correlation Heatmap", "Box Plot"]
    )
    
    # Dynamic axis selection
    x_axis = st.selectbox("X-Axis", df.columns)
    y_axis = st.selectbox("Y-Axis", df.columns) if chart_type != "Histogram" else None
    
    # Generate plot
    fig, ax = plt.subplots()
    if chart_type == "Scatter Plot":
        df.plot(kind='scatter', x=x_axis, y=y_axis, ax=ax)
    elif chart_type == "Histogram":
        df[x_axis].plot(kind='hist', ax=ax)
    elif chart_type == "Correlation Heatmap":
        corr = df.corr()
        ax.matshow(corr)
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
    else:  # Box Plot
        df.plot(kind='box', ax=ax)
    
    st.pyplot(fig)

# --- Tab 3: Data Cleaning ---
def cleaning_tab():
    st.header("🧹 Basic Data Cleaning")
    
    if st.session_state.df is None:
        st.warning("Upload data in the AutoML tab first!")
        return
    
    df = st.session_state.df
    
    # Missing values analysis
    st.subheader("Missing Values")
    missing_data = df.isnull().sum()
    st.bar_chart(missing_data)
    
    # Cleaning options
    st.subheader("Cleaning Tools")
    if st.button("Remove Rows with Missing Values"):
        df_cleaned = df.dropna()
        st.session_state.df = df_cleaned
        st.success(f"Removed {len(df) - len(df_cleaned)} rows")
    
    if st.button("Fill Missing with Mean"):
        df_cleaned = df.fillna(df.mean())
        st.session_state.df = df_cleaned
        st.success("Applied mean imputation")

# --- Main App ---
st.title("📊 DataLab Light")
tab1, tab2, tab3 = st.tabs(["AutoML", "Visualization", "Data Cleaning"])

with tab1:
    automl_tab()

with tab2:
    visualization_tab()

with tab3:
    cleaning_tab()

st.markdown("---")
st.caption("Note: This lightweight version uses only core Python data science libraries")
