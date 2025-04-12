import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from pycaret.classification import setup, compare_models
import missingno as msno

# Page Config
st.set_page_config(
    page_title="DataLab Pro",
    page_icon="📊",
    layout="wide"
)

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
        st.success("Data loaded successfully!")
        
        # Target selection
        target = st.selectbox("Select Target Column", df.columns)
        
        if st.button("Train Models"):
            with st.spinner("Running AutoML..."):
                # PyCaret setup
                setup(df, target=target, silent=True)
                best_model = compare_models()
                st.success(f"Best Model: {best_model.__class__.__name__}")
                
                # Feature importance
                try:
                    importances = best_model.feature_importances_
                    feat_imp = pd.DataFrame({
                        "Feature": df.drop(target, axis=1).columns,
                        "Importance": importances
                    }).sort_values("Importance", ascending=False)
                    
                    st.plotly_chart(px.bar(feat_imp, x="Feature", y="Importance", title="Feature Importance"))
                except:
                    st.warning("Feature importance not available for this model.")

# --- Tab 2: Data Visualization ---
def visualization_tab():
    st.header("📈 Interactive Visualizations")
    
    if st.session_state.df is None:
        st.warning("Upload data in the AutoML tab first!")
        return
    
    df = st.session_state.df
    
    # Chart type selection
    chart_type = st.selectbox(
        "Choose Chart Type",
        ["Scatter Plot", "Histogram", "Heatmap", "Box Plot"]
    )
    
    # Dynamic axis selection
    x_axis = st.selectbox("X-Axis", df.columns)
    y_axis = st.selectbox("Y-Axis", df.columns) if chart_type != "Histogram" else None
    
    # Generate plot
    if chart_type == "Scatter Plot":
        fig = px.scatter(df, x=x_axis, y=y_axis, color=df.columns[0])
    elif chart_type == "Histogram":
        fig = px.histogram(df, x=x_axis)
    elif chart_type == "Heatmap":
        fig = px.imshow(df.corr())
    else:  # Box Plot
        fig = px.box(df, x=x_axis, y=y_axis)
    
    st.plotly_chart(fig, use_container_width=True)

# --- Tab 3: Data Cleaning ---
def cleaning_tab():
    st.header("🧹 Data Cleaning")
    
    if st.session_state.df is None:
        st.warning("Upload data in the AutoML tab first!")
        return
    
    df = st.session_state.df
    
    # Missing values matrix
    st.subheader("Missing Data Analysis")
    msno.matrix(df)
    st.pyplot()
    
    # Cleaning options
    st.subheader("Auto-Fix Tools")
    if st.button("Remove Rows with Missing Values"):
        df_cleaned = df.dropna()
        st.session_state.df = df_cleaned
        st.success(f"Removed {len(df) - len(df_cleaned)} rows")
    
    if st.button("Fill Missing Values with Median"):
        df_cleaned = df.fillna(df.median())
        st.session_state.df = df_cleaned
        st.success("Applied median imputation")

# --- Main App ---
st.title("📊 DataLab Pro")
tab1, tab2, tab3 = st.tabs(["AutoML", "Visualization", "Data Cleaning"])

with tab1:
    automl_tab()

with tab2:
    visualization_tab()

with tab3:
    cleaning_tab()
