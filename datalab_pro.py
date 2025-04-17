import streamlit as st import pandas as pd import numpy as np import matplotlib matplotlib.use('Agg') import matplotlib.pyplot as plt import shap import joblib import io from sklearn.model_selection import train_test_split from sklearn.ensemble import RandomForestClassifier from sklearn.metrics import accuracy_score, confusion_matrix, classification_report from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="DataLab Pro", page_icon="📊", layout="wide") st.markdown(""" <style> #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;} </style> """, unsafe_allow_html=True)

if 'df' not in st.session_state: st.session_state.df = None if 'model' not in st.session_state: st.session_state.model = None

--- TAB 1: AutoML ---

def automl_tab(): st.header("🔮 AutoML") uploaded_file = st.file_uploader("Upload CSV", type=["csv"]) use_sample = st.button("📁 Use Sample Data")

if uploaded_file or use_sample:
    try:
        if use_sample:
            df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
            st.info("Using sample Iris dataset.")
        else:
            df = pd.read_csv(uploaded_file)

        st.session_state.df = df
        st.write("Data Preview:")
        st.dataframe(df.head())

        target = st.selectbox("Select Target Column", df.columns)

        if st.button("Train Model"):
            with st.spinner("Training model..."):
                X = pd.get_dummies(df.drop(target, axis=1))
                y = df[target]
                le = LabelEncoder()
                y = le.fit_transform(y)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                st.session_state.model = model
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

                st.subheader("Classification Report")
                st.text(classification_report(y_test, y_pred))

                st.download_button("📥 Download Trained Model", data=joblib.dump(model, 'model.pkl'), file_name="model.pkl")

    except Exception:
        st.error("❌ We couldn't read your file. Please make sure it's a valid CSV or try the sample dataset.")

--- TAB 2: SHAP Explainability ---

def explainability_tab(): st.header("🧠 Explainable AI with SHAP") if st.session_state.df is None or st.session_state.model is None: st.warning("Please upload data and train a model in AutoML tab first.") return

df = st.session_state.df
model = st.session_state.model
X = pd.get_dummies(df.drop(columns=df.columns[-1]))

with st.spinner("Generating SHAP values..."):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.subheader("Feature Importance")
    shap.summary_plot(shap_values, X, plot_type="bar")
    st.pyplot(bbox_inches='tight')

--- TAB 3: Data Visualization ---

def visualization_tab(): st.header("📊 Data Visualization") if st.session_state.df is None: st.warning("Please upload a CSV in the AutoML tab first.") return

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

--- TAB 4: Data Cleaning ---

def cleaning_tab(): st.header("🧹 Data Cleaning") if st.session_state.df is None: st.warning("Upload a CSV in the AutoML tab.") return

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

--- MAIN UI ---

st.title("📊 DataLab Pro") tab1, tab2, tab3, tab4 = st.tabs(["AutoML", "Explainability", "Visualization", "Data Cleaning"])

with tab1: automl_tab() with tab2: explainability_tab() with tab3: visualization_tab() with tab4: cleaning_tab()

