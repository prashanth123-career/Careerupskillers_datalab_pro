import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Streamlit compatibility
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

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

# --- Main App ---
st.set_page_config(
    page_title="DataLab Pro",
    page_icon="📊",
    layout="wide"
)

st.title("📊 DataLab Pro")
tab1, tab2, tab3 = st.tabs(["AutoML", "Visualization", "Data Cleaning"])

with tab1:
    automl_tab()

with tab2:
    visualization_tab()

with tab3:
    cleaning_tab()
