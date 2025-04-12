import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Replaces Plotly
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
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        
        target = st.selectbox("Select Target Column", df.columns)
        
        if st.button("Train Model"):
            X = df.drop(target, axis=1)
            y = df[target]
            X = pd.get_dummies(X)  # Simple encoding
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            
            # Results
            st.success(f"Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2f}")
            
            # Confusion Matrix
            fig, ax = plt.subplots()
            cm = confusion_matrix(y_test, model.predict(X_test))
            ax.matshow(cm, cmap=plt.cm.Blues)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i, j], ha='center', va='center')
            st.pyplot(fig)

# --- Tab 2: Visualization ---
def visualization_tab():
    if st.session_state.df is None:
        st.warning("Upload data first!")
        return
    
    df = st.session_state.df
    chart_type = st.selectbox("Chart Type", ["Histogram", "Scatter", "Boxplot"])
    col = st.selectbox("Column", df.columns)
    
    fig, ax = plt.subplots()
    if chart_type == "Histogram":
        df[col].hist(ax=ax)
    elif chart_type == "Scatter":
        y_col = st.selectbox("Y-Axis", df.columns)
        df.plot.scatter(x=col, y=y_col, ax=ax)
    else:
        df.boxplot(column=col, ax=ax)
    st.pyplot(fig)

# --- Tab 3: Data Cleaning ---
def cleaning_tab():
    if st.session_state.df is None:
        st.warning("Upload data first!")
        return
    
    df = st.session_state.df
    st.write("Missing Values:", df.isnull().sum())
    
    if st.button("Remove NA Rows"):
        st.session_state.df = df.dropna()
        st.success(f"Removed {len(df) - len(st.session_state.df)} rows")

# --- App Layout ---
st.title("📊 DataLab Pro")
tab1, tab2, tab3 = st.tabs(["AutoML", "Visualization", "Data Cleaning"])
with tab1: automl_tab()
with tab2: visualization_tab()
with tab3: cleaning_tab()
