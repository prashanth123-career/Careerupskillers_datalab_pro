import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import io
import time
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                            roc_auc_score, precision_recall_curve, roc_curve)
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from yellowbrick.classifier import ROCAUC, ClassificationReport, ConfusionMatrix
import sweetviz as sv
from pandas_profiling import ProfileReport
import missingno as msno
import base64
from fpdf import FPDF
import tempfile
import os

# Configure Streamlit
st.set_page_config(page_title="DataLab Pro", page_icon="📊", layout="wide")
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .css-18e3th9 {padding-top: 0rem;}
        .css-1d391kg {padding-top: 3.5rem;}
        .reportview-container .main .block-container {padding-top: 2rem;}
        .stProgress > div > div > div > div {background-image: linear-gradient(to right, #6a11cb, #2575fc);}
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'train_history' not in st.session_state:
    st.session_state.train_history = []
if 'best_params' not in st.session_state:
    st.session_state.best_params = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None

# --- Helper Functions ---
def create_download_link(val, filename):
    """Generate download link for files"""
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}">Download {filename}</a>'

def generate_pdf_report():
    """Generate a comprehensive PDF report and return as bytes for download"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Add title
    pdf.cell(200, 10, txt="DataLab Pro Analysis Report", ln=1, align='C')
    pdf.cell(200, 10, txt=f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1, align='C')
    pdf.ln(10)
    
    # Add dataset info
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(200, 10, txt="Dataset Information", ln=1)
    pdf.set_font("Arial", size=10)
    if hasattr(st.session_state, "df") and st.session_state.df is not None:
        pdf.cell(200, 10, txt=f"Shape: {st.session_state.df.shape}", ln=1)
        pdf.cell(200, 10, txt=f"Columns: {', '.join(st.session_state.df.columns)}", ln=1)
    else:
        pdf.cell(200, 10, txt="No dataset loaded.", ln=1)

    # Add model info
    if hasattr(st.session_state, "model") and st.session_state.model:
        pdf.ln(5)
        pdf.set_font("Arial", 'B', size=12)
        pdf.cell(200, 10, txt="Model Information", ln=1)
        pdf.set_font("Arial", size=10)
        pdf.cell(200, 10, txt=f"Model type: {type(st.session_state.model).__name__}", ln=1)
        if hasattr(st.session_state, "best_params") and st.session_state.best_params:
            pdf.cell(200, 10, txt=f"Best parameters: {str(st.session_state.best_params)}", ln=1)
    else:
        pdf.ln(5)
        pdf.set_font("Arial", 'B', size=12)
        pdf.cell(200, 10, txt="Model Information", ln=1)
        pdf.set_font("Arial", size=10)
        pdf.cell(200, 10, txt="No model trained.", ln=1)
    
    # Output PDF to bytes
    pdf_buffer = io.BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

# Example usage in Streamlit:
if st.button("Generate PDF Report"):
    pdf_bytes = generate_pdf_report()
    st.download_button(
        label="Download PDF Report",
        data=pdf_bytes,
        file_name="datalab_pro_report.pdf",
        mime="application/pdf"
    )
def generate_sweetviz_report():
    """Generate EDA report using Sweetviz"""
    report = sv.analyze(st.session_state.df)
    report_file = "sweetviz_report.html"
    report.show_html(report_file, open_browser=False)
    with open(report_file, "rb") as f:
        html = f.read()
    os.unlink(report_file)
    return html

def generate_pandas_profiling():
    """Generate EDA report using pandas profiling"""
    profile = ProfileReport(st.session_state.df, explorative=True)
    profile_file = "profile_report.html"
    profile.to_file(profile_file)
    with open(profile_file, "rb") as f:
        html = f.read()
    os.unlink(profile_file)
    return html

def get_data_quality_suggestions(df):
    """Generate data quality suggestions"""
    suggestions = []
    
    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        suggestions.append(f"⚠️ Missing values detected in {missing[missing > 0].count()} columns. Consider imputation or removal.")
    
    # Duplicates
    if df.duplicated().sum() > 0:
        suggestions.append(f"⚠️ {df.duplicated().sum()} duplicate rows found. Consider removing duplicates.")
    
    # Zero variance columns
    nunique = df.nunique()
    if (nunique == 1).sum() > 0:
        suggestions.append(f"⚠️ {(nunique == 1).sum()} constant columns detected. These may not be useful for modeling.")
    
    # High cardinality
    high_cardinality = nunique[nunique > 50].index.tolist()
    if high_cardinality:
        suggestions.append(f"⚠️ High cardinality features detected: {', '.join(high_cardinality)}. Consider feature engineering.")
    
    # Numeric outliers
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 0:
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = ((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr)))
            if outliers.sum() > 0:
                suggestions.append(f"⚠️ {outliers.sum()} potential outliers detected in {col}. Consider transformation or removal.")
    
    if not suggestions:
        suggestions.append("✅ Data quality looks good! No major issues detected.")
    
    return suggestions

def get_model_suggestions(model, X_train, y_train):
    """Generate model improvement suggestions"""
    suggestions = []
    
    # Feature importance analysis
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        zero_importance = sum(importances == 0)
        if zero_importance > 0:
            suggestions.append(f"🔍 {zero_importance} features have zero importance. Consider removing them.")
    
    # Class imbalance check
    class_counts = pd.Series(y_train).value_counts()
    if len(class_counts) > 1 and (class_counts.min() / class_counts.max()) < 0.3:
        suggestions.append("⚖️ Class imbalance detected. Consider using class weights, oversampling, or different evaluation metrics.")
    
    # Model-specific suggestions
    if isinstance(model, RandomForestClassifier):
        suggestions.append("🌲 For Random Forest, consider tuning max_depth and n_estimators further.")
    elif isinstance(model, GradientBoostingClassifier):
        suggestions.append("📈 For Gradient Boosting, consider tuning learning_rate and n_estimators.")
    elif isinstance(model, LogisticRegression):
        suggestions.append("📉 For Logistic Regression, consider trying different regularization strengths (C parameter).")
    
    if not suggestions:
        suggestions.append("✅ Model looks good! Consider trying more complex models if you need better performance.")
    
    return suggestions

# --- TAB 1: AutoML ---
def automl_tab():
    st.header("🔮 AutoML Pro")
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        use_sample = st.checkbox("Use Sample Data (Iris Dataset)")
    
    with col2:
        st.markdown("### Premium Features")
        st.markdown("""
        - Advanced model tuning with GridSearch
        - Multiple algorithm support
        - Automated feature engineering
        - Model comparison
        - Detailed reports
        """)
    
    if uploaded_file or use_sample:
        try:
            if use_sample:
                df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
                st.info("Using sample Iris dataset.")
            else:
                df = pd.read_csv(uploaded_file)
            
            st.session_state.df = df
            
            with st.expander("🔍 Data Preview", expanded=True):
                st.dataframe(df.head())
            
            # Data quality suggestions
            st.subheader("📋 Data Quality Report")
            suggestions = get_data_quality_suggestions(df)
            for suggestion in suggestions:
                st.write(suggestion)
            
            # Data cleaning options
            st.subheader("🧹 Data Cleaning Options")
            cleaning_options = st.multiselect(
                "Select cleaning operations:",
                ["Drop rows with missing values", 
                 "Fill numeric missing with mean", 
                 "Fill categorical missing with mode",
                 "Remove duplicate rows",
                 "Remove constant columns"]
            )
            
            if st.button("Apply Cleaning"):
                df_clean = df.copy()
                
                if "Drop rows with missing values" in cleaning_options:
                    df_clean = df_clean.dropna()
                
                if "Fill numeric missing with mean" in cleaning_options:
                    num_cols = df_clean.select_dtypes(include=np.number).columns
                    df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].mean())
                
                if "Fill categorical missing with mode" in cleaning_options:
                    cat_cols = df_clean.select_dtypes(exclude=np.number).columns
                    for col in cat_cols:
                        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
                
                if "Remove duplicate rows" in cleaning_options:
                    df_clean = df_clean.drop_duplicates()
                
                if "Remove constant columns" in cleaning_options:
                    nunique = df_clean.nunique()
                    constant_cols = nunique[nunique == 1].index.tolist()
                    df_clean = df_clean.drop(columns=constant_cols)
                
                st.session_state.df = df_clean
                st.success("Data cleaning applied successfully!")
                st.dataframe(df_clean.head())
            
            # Feature engineering
            st.subheader("⚙️ Feature Engineering")
            target = st.selectbox("Select Target Column", df.columns)
            
            if st.button("Auto Feature Engineering"):
                with st.spinner("Performing feature engineering..."):
                    X = st.session_state.df.drop(target, axis=1)
                    y = st.session_state.df[target]
                    
                    # Preprocessing pipeline
                    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
                    categorical_features = X.select_dtypes(include=['object', 'category']).columns
                    
                    numeric_transformer = Pipeline(steps=[
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())])
                    
                    categorical_transformer = Pipeline(steps=[
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
                    
                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('num', numeric_transformer, numeric_features),
                            ('cat', categorical_transformer, categorical_features)])
                    
                    st.session_state.preprocessor = preprocessor
                    X_processed = preprocessor.fit_transform(X)
                    
                    # Feature selection
                    selector = SelectKBest(f_classif, k='all')
                    selector.fit(X_processed, y)
                    
                    # Get feature names
                    numeric_features_out = numeric_features
                    if len(categorical_features) > 0:
                        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
                        categorical_features_out = cat_encoder.get_feature_names_out(categorical_features)
                        all_features = np.concatenate([numeric_features_out, categorical_features_out])
                    else:
                        all_features = numeric_features_out
                    
                    st.session_state.feature_names = all_features
                    
                    # Display feature importance
                    fig, ax = plt.subplots(figsize=(10, 6))
                    pd.Series(selector.scores_, index=all_features).sort_values().plot.barh(ax=ax)
                    ax.set_title("Feature Importance Scores")
                    st.pyplot(fig)
                    
                    st.success("Feature engineering completed!")
            
            # Model training
            st.subheader("🤖 Model Training")
            
            model_options = {
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "SVM": SVC(probability=True),
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "K-Nearest Neighbors": KNeighborsClassifier()
            }
            
            selected_model = st.selectbox("Select Model", list(model_options.keys()))
            
            advanced_tuning = st.checkbox("Enable Advanced Hyperparameter Tuning (Slower)")
            
            if st.button("Train Model"):
                if st.session_state.df is None:
                    st.warning("Please process your data first.")
                    return
                
                with st.spinner("Training model..."):
                    X = st.session_state.df.drop(target, axis=1)
                    y = st.session_state.df[target]
                    
                    # Encode target if categorical
                    if y.dtype == 'object':
                        le = LabelEncoder()
                        y = le.fit_transform(y)
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42)
                    
                    # Preprocess data
                    if st.session_state.preprocessor is None:
                        # If no preprocessing done yet, create basic preprocessor
                        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
                        categorical_features = X.select_dtypes(include=['object', 'category']).columns
                        
                        numeric_transformer = Pipeline(steps=[
                            ('imputer', SimpleImputer(strategy='median')),
                            ('scaler', StandardScaler())])
                        
                        categorical_transformer = Pipeline(steps=[
                            ('imputer', SimpleImputer(strategy='most_frequent')),
                            ('onehot', OneHotEncoder(handle_unknown='ignore'))])
                        
                        preprocessor = ColumnTransformer(
                            transformers=[
                                ('num', numeric_transformer, numeric_features),
                                ('cat', categorical_transformer, categorical_features)])
                        
                        st.session_state.preprocessor = preprocessor
                    
                    # Create pipeline
                    model = model_options[selected_model]
                    pipeline = Pipeline(steps=[
                        ('preprocessor', st.session_state.preprocessor),
                        ('classifier', model)])
                    
                    # Hyperparameter tuning
                    if advanced_tuning:
                        param_grids = {
                            "Random Forest": {
                                'classifier__n_estimators': [50, 100, 200],
                                'classifier__max_depth': [None, 5, 10],
                                'classifier__min_samples_split': [2, 5]
                            },
                            "Gradient Boosting": {
                                'classifier__n_estimators': [50, 100],
                                'classifier__learning_rate': [0.01, 0.1, 0.5],
                                'classifier__max_depth': [3, 5]
                            },
                            "SVM": {
                                'classifier__C': [0.1, 1, 10],
                                'classifier__kernel': ['linear', 'rbf']
                            },
                            "Logistic Regression": {
                                'classifier__C': [0.1, 1, 10],
                                'classifier__penalty': ['l2']
                            },
                            "K-Nearest Neighbors": {
                                'classifier__n_neighbors': [3, 5, 7],
                                'classifier__weights': ['uniform', 'distance']
                            }
                        }
                        
                        grid_search = GridSearchCV(
                            pipeline,
                            param_grids[selected_model],
                            cv=5,
                            n_jobs=-1,
                            verbose=1)
                        
                        grid_search.fit(X_train, y_train)
                        best_model = grid_search.best_estimator_
                        st.session_state.best_params = grid_search.best_params_
                    else:
                        pipeline.fit(X_train, y_train)
                        best_model = pipeline
                    
                    st.session_state.model = best_model
                    
                    # Evaluate model
                    y_pred = best_model.predict(X_test)
                    y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
                    
                    acc = accuracy_score(y_test, y_pred)
                    st.success(f"Model trained successfully with accuracy: {acc:.2f}")
                    
                    # Save training history
                    st.session_state.train_history.append({
                        'timestamp': datetime.now(),
                        'model_type': selected_model,
                        'accuracy': acc,
                        'params': st.session_state.best_params if advanced_tuning else 'default'
                    })
                    
                    # Display metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Confusion Matrix")
                        fig, ax = plt.subplots()
                        cm = confusion_matrix(y_test, y_pred)
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        st.pyplot(fig)
                    
                    with col2:
                        st.subheader("Classification Report")
                        report = classification_report(y_test, y_pred, output_dict=True)
                        st.table(pd.DataFrame(report).transpose())
                    
                    # ROC Curve if probabilities available
                    if y_proba is not None and len(np.unique(y_test)) == 2:
                        st.subheader("ROC Curve")
                        fpr, tpr, _ = roc_curve(y_test, y_proba)
                        roc_auc = roc_auc_score(y_test, y_proba)
                        
                        fig, ax = plt.subplots()
                        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                                label=f'ROC curve (area = {roc_auc:.2f})')
                        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                        ax.set_xlim([0.0, 1.0])
                        ax.set_ylim([0.0, 1.05])
                        ax.set_xlabel('False Positive Rate')
                        ax.set_ylabel('True Positive Rate')
                        ax.set_title('Receiver Operating Characteristic')
                        ax.legend(loc="lower right")
                        st.pyplot(fig)
                    
                    # Model suggestions
                    st.subheader("🔍 Model Improvement Suggestions")
                    suggestions = get_model_suggestions(best_model.named_steps['classifier'], 
                                                      st.session_state.preprocessor.transform(X_train), 
                                                      y_train)
                    for suggestion in suggestions:
                        st.write(suggestion)
            
            # Report generation
            st.subheader("📄 Generate Reports")
            report_type = st.selectbox("Select Report Type", 
                                     ["PDF Summary", "Sweetviz EDA", "Pandas Profiling"])
            
            if st.button("Generate Report"):
                with st.spinner(f"Generating {report_type} report..."):
                    if report_type == "PDF Summary":
                        report_bytes = generate_pdf_report()
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=report_bytes,
                            file_name="datalab_report.pdf",
                            mime="application/pdf"
                        )
                    elif report_type == "Sweetviz EDA":
                        report_html = generate_sweetviz_report()
                        st.download_button(
                            label="📥 Download Sweetviz Report",
                            data=report_html,
                            file_name="sweetviz_report.html",
                            mime="text/html"
                        )
                    elif report_type == "Pandas Profiling":
                        report_html = generate_pandas_profiling()
                        st.download_button(
                            label="📥 Download Pandas Profiling",
                            data=report_html,
                            file_name="pandas_profiling.html",
                            mime="text/html"
                        )
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.error("We couldn't process your file. Please check the format or try the sample dataset.")

# --- TAB 2: SHAP Explainability ---
def explainability_tab():
    st.header("🧠 Explainable AI with SHAP")
    
    if st.session_state.df is None or st.session_state.model is None:
        st.warning("Please upload data and train a model in AutoML tab first.")
        return
    
    df = st.session_state.df
    model = st.session_state.model
    target = st.selectbox("Select Target Column (for reference)", df.columns)
    
    X = df.drop(target, axis=1)
    explainer_type = st.selectbox("SHAP Explainer Type", 
                                 ["TreeExplainer (for tree models)", 
                                  "KernelExplainer (any model)"])
    
    sample_size = st.slider("Sample Size for SHAP Analysis", 10, 500, 100)
    
    if st.button("Run SHAP Analysis"):
        with st.spinner("Calculating SHAP values..."):
            try:
                # Preprocess the data
                X_processed = st.session_state.preprocessor.transform(X)
                
                if isinstance(X_processed, np.ndarray):
                    X_processed = pd.DataFrame(X_processed, columns=st.session_state.feature_names)
                
                # Sample the data for faster computation
                if len(X_processed) > sample_size:
                    X_sampled = X_processed.sample(sample_size, random_state=42)
                else:
                    X_sampled = X_processed
                
                # Create explainer
                if "TreeExplainer" in explainer_type and hasattr(model.named_steps['classifier'], 'tree_'):
                    explainer = shap.TreeExplainer(model.named_steps['classifier'])
                    shap_values = explainer.shap_values(X_sampled)
                else:
                    def model_predict(X):
                        return model.predict_proba(X)
                    
                    explainer = shap.KernelExplainer(model_predict, shap.kmeans(X_processed, 5))
                    shap_values = explainer.shap_values(X_sampled)
                
                # Visualizations
                st.subheader("Global Feature Importance")
                fig, ax = plt.subplots()
                shap.summary_plot(shap_values, X_sampled, plot_type="bar", show=False)
                st.pyplot(fig)
                
                st.subheader("SHAP Summary Plot")
                fig, ax = plt.subplots()
                shap.summary_plot(shap_values, X_sampled, show=False)
                st.pyplot(fig)
                
                # Individual prediction explanations
                st.subheader("Individual Prediction Explanations")
                sample_idx = st.selectbox("Select sample to explain", range(len(X_sampled)))
                
                fig, ax = plt.subplots()
                shap.force_plot(explainer.expected_value, 
                               shap_values[sample_idx,:], 
                               X_sampled.iloc[sample_idx,:],
                               matplotlib=True, show=False)
                st.pyplot(fig)
                
                # Dependence plots
                st.subheader("Feature Dependence Plots")
                feature = st.selectbox("Select feature for dependence plot", X_sampled.columns)
                
                fig, ax = plt.subplots()
                shap.dependence_plot(feature, shap_values, X_sampled, show=False)
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"SHAP analysis failed: {str(e)}")

# --- TAB 3: Data Visualization ---
def visualization_tab():
    st.header("📊 Advanced Data Visualization")
    
    if st.session_state.df is None:
        st.warning("Please upload a CSV in the AutoML tab first.")
        return
    
    df = st.session_state.df
    
    st.subheader("Interactive Visualizations")
    chart_type = st.selectbox("Choose Chart Type", 
                             ["Histogram", 
                              "Scatter Plot", 
                              "Box Plot", 
                              "Violin Plot",
                              "Correlation Heatmap",
                              "Pair Plot",
                              "Pie Chart"])
    
    if chart_type in ["Histogram", "Box Plot", "Violin Plot", "Pie Chart"]:
        col = st.selectbox("Select Column", df.columns)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if chart_type == "Histogram":
            if pd.api.types.is_numeric_dtype(df[col]):
                sns.histplot(df[col], kde=True, ax=ax)
                ax.set_title(f"Distribution of {col}")
            else:
                st.warning("Please select a numeric column for histogram.")
        
        elif chart_type == "Box Plot":
            if pd.api.types.is_numeric_dtype(df[col]):
                sns.boxplot(x=df[col], ax=ax)
                ax.set_title(f"Box Plot of {col}")
            else:
                st.warning("Please select a numeric column for box plot.")
        
        elif chart_type == "Violin Plot":
            if pd.api.types.is_numeric_dtype(df[col]):
                sns.violinplot(x=df[col], ax=ax)
                ax.set_title(f"Violin Plot of {col}")
            else:
                st.warning("Please select a numeric column for violin plot.")
        
        elif chart_type == "Pie Chart":
            if not pd.api.types.is_numeric_dtype(df[col]):
                value_counts = df[col].value_counts()
                ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%')
                ax.set_title(f"Distribution of {col}")
            else:
                st.warning("Please select a categorical column for pie chart.")
        
        st.pyplot(fig)
    
    elif chart_type == "Scatter Plot":
        col1 = st.selectbox("X-Axis Column", df.columns)
        col2 = st.selectbox("Y-Axis Column", df.columns)
        hue_col = st.selectbox("Hue Column (optional)", [None] + list(df.columns))
        
        if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(data=df, x=col1, y=col2, hue=hue_col, ax=ax)
            ax.set_title(f"{col1} vs {col2}")
            st.pyplot(fig)
        else:
            st.warning("Both columns should be numeric for scatter plot.")
    
    elif chart_type == "Correlation Heatmap":
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) >= 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)
        else:
            st.warning("Need at least 2 numeric columns for correlation heatmap.")
    
    elif chart_type == "Pair Plot":
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) >= 2:
            hue_col = st.selectbox("Hue Column for Pair Plot", [None] + list(df.columns))
            fig = sns.pairplot(df, vars=numeric_cols[:5], hue=hue_col)  # Limit to 5 cols for performance
            st.pyplot(fig)
        else:
            st.warning("Need at least 2 numeric columns for pair plot.")
    
    # Missing data visualization
    st.subheader("Missing Data Visualization")
    if st.checkbox("Show Missing Data Pattern"):
        fig, ax = plt.subplots(figsize=(10, 6))
        msno.matrix(df, ax=ax)
        st.pyplot(fig)

# --- TAB 4: Data Cleaning ---
def cleaning_tab():
    st.header("🧹 Advanced Data Cleaning")
    
    if st.session_state.df is None:
        st.warning("Upload a CSV in the AutoML tab.")
        return
    
    df = st.session_state.df
    
    st.subheader("Data Quality Dashboard")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Rows", len(df))
    
    with col2:
        st.metric("Total Columns", len(df.columns))
    
    with col3:
        missing = df.isnull().sum().sum()
        st.metric("Missing Values", missing)
    
    st.subheader("Missing Values Analysis")
    st.bar_chart(df.isnull().sum())
    
    st.subheader("Data Cleaning Operations")
    
    cleaning_options = st.multiselect(
        "Select cleaning operations to apply:",
        ["Drop rows with missing values", 
         "Fill numeric missing with median",
         "Fill numeric missing with mean",
         "Fill categorical missing with mode",
         "Remove duplicate rows",
         "Remove constant columns",
         "Convert text to lowercase",
         "Remove special characters from text"]
    )
    
    if st.button("Apply Cleaning Operations"):
        df_clean = df.copy()
        
        if "Drop rows with missing values" in cleaning_options:
            df_clean = df_clean.dropna()
        
        if "Fill numeric missing with median" in cleaning_options:
            num_cols = df_clean.select_dtypes(include=np.number).columns
            df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].median())
        
        if "Fill numeric missing with mean" in cleaning_options:
            num_cols = df_clean.select_dtypes(include=np.number).columns
            df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].mean())
        
        if "Fill categorical missing with mode" in cleaning_options:
            cat_cols = df_clean.select_dtypes(exclude=np.number).columns
            for col in cat_cols:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        
        if "Remove duplicate rows" in cleaning_options:
            df_clean = df_clean.drop_duplicates()
        
        if "Remove constant columns" in cleaning_options:
            nunique = df_clean.nunique()
            constant_cols = nunique[nunique == 1].index.tolist()
            df_clean = df_clean.drop(columns=constant_cols)
        
        if "Convert text to lowercase" in cleaning_options:
            text_cols = df_clean.select_dtypes(include=['object']).columns
            for col in text_cols:
                df_clean[col] = df_clean[col].str.lower()
        
        if "Remove special characters from text" in cleaning_options:
            text_cols = df_clean.select_dtypes(include=['object']).columns
            for col in text_cols:
                df_clean[col] = df_clean[col].str.replace(r'[^\w\s]', '', regex=True)
        
        st.session_state.df = df_clean
        st.success("Data cleaning operations applied successfully!")
        st.dataframe(df_clean.head())
    
    st.subheader("Column-Specific Operations")
    col_to_clean = st.selectbox("Select column to clean", df.columns)
    
    if pd.api.types.is_numeric_dtype(df[col_to_clean]):
        operation = st.selectbox("Select operation", 
                               ["Log transform", 
                                "Square root transform",
                                "Standardize (z-score)",
                                "Normalize (0-1)",
                                "Remove outliers"])
        
        if st.button(f"Apply to {col_to_clean}"):
            df_clean = df.copy()
            
            if operation == "Log transform":
                df_clean[col_to_clean] = np.log1p(df_clean[col_to_clean])
            
            elif operation == "Square root transform":
                df_clean[col_to_clean] = np.sqrt(df_clean[col_to_clean])
            
            elif operation == "Standardize (z-score)":
                mean = df_clean[col_to_clean].mean()
                std = df_clean[col_to_clean].std()
                df_clean[col_to_clean] = (df_clean[col_to_clean] - mean) / std
            
            elif operation == "Normalize (0-1)":
                min_val = df_clean[col_to_clean].min()
                max_val = df_clean[col_to_clean].max()
                df_clean[col_to_clean] = (df_clean[col_to_clean] - min_val) / (max_val - min_val)
            
            elif operation == "Remove outliers":
                q1 = df_clean[col_to_clean].quantile(0.25)
                q3 = df_clean[col_to_clean].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                df_clean = df_clean[(df_clean[col_to_clean] >= lower_bound) & 
                                   (df_clean[col_to_clean] <= upper_bound)]
            
            st.session_state.df = df_clean
            st.success(f"Operation '{operation}' applied to {col_to_clean}!")
            st.dataframe(df_clean.head())
    
    else:
        operation = st.selectbox("Select operation", 
                               ["One-hot encode", 
                                "Label encode",
                                "Extract text features",
                                "Clean whitespace"])
        
        if st.button(f"Apply to {col_to_clean}"):
            df_clean = df.copy()
            
            if operation == "One-hot encode":
                dummies = pd.get_dummies(df_clean[col_to_clean], prefix=col_to_clean)
                df_clean = pd.concat([df_clean.drop(col_to_clean, axis=1), dummies], axis=1)
            
            elif operation == "Label encode":
                le = LabelEncoder()
                df_clean[col_to_clean] = le.fit_transform(df_clean[col_to_clean])
            
            elif operation == "Extract text features":
                df_clean[f"{col_to_clean}_length"] = df_clean[col_to_clean].str.len()
                df_clean[f"{col_to_clean}_word_count"] = df_clean[col_to_clean].str.split().str.len()
            
            elif operation == "Clean whitespace":
                df_clean[col_to_clean] = df_clean[col_to_clean].str.strip()
            
            st.session_state.df = df_clean
            st.success(f"Operation '{operation}' applied to {col_to_clean}!")
            st.dataframe(df_clean.head())

# --- TAB 5: Model History ---
def history_tab():
    st.header("🕰️ Model Training History")
    
    if not st.session_state.train_history:
        st.warning("No training history available. Train some models first!")
        return
    
    st.subheader("Training Sessions")
    
    for i, session in enumerate(st.session_state.train_history):
        with st.expander(f"Session {i+1} - {session['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Model Type:** {session['model_type']}")
                st.write(f"**Accuracy:** {session['accuracy']:.2f}")
            
            with col2:
                st.write("**Parameters:**")
                st.json(session['params'])
    
    st.subheader("Performance Trend")
    if len(st.session_state.train_history) > 1:
        history_df = pd.DataFrame(st.session_state.train_history)
        history_df['timestamp'] = history_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=history_df, x='timestamp', y='accuracy', 
                    hue='model_type', marker='o', ax=ax)
        ax.set_title("Model Accuracy Over Time")
        ax.set_xlabel("Training Session")
        ax.set_ylabel("Accuracy")
        plt.xticks(rotation=45)
        st.pyplot(fig)

# --- MAIN UI ---
st.title("💎 DataLab Pro - Premium Edition")

# Add premium branding
st.sidebar.image("https://via.placeholder.com/300x100?text=DataLab+Pro", use_column_width=True)
st.sidebar.markdown("""
    ### Premium Features
    - Advanced AutoML with hyperparameter tuning
    - Comprehensive data cleaning toolkit
    - Interactive visualizations
    - SHAP explainability
    - Multiple report formats
    - Model history tracking
""")

st.sidebar.markdown("""
    ### Subscription Status
    🔒 **Premium Edition**  
    Valid until: 2024-12-31  
    [Manage Subscription](#)
""")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["AutoML", "Explainability", "Visualization", "Data Cleaning", "History"])

with tab1:
    automl_tab()
with tab2:
    explainability_tab()
with tab3:
    visualization_tab()
with tab4:
    cleaning_tab()
with tab5:
    history_tab()

# Add footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: gray;">
        <p>DataLab Pro Premium Edition © 2023 | All Rights Reserved</p>
        <p>Need help? Contact <a href="mailto:support@datalabpro.com">support@datalabpro.com</a></p>
    </div>
""", unsafe_allow_html=True)
