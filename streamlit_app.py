# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from io import BytesIO
from sklearn.preprocessing import LabelEncoder

# ============ Load Model ============
model = joblib.load("model/fraud_detection_xgb_model.pkl")

# ============ Sidebar ============

st.set_page_config(layout="wide", page_title="Fraud Detection System")
st.sidebar.title("🛡️ Fraud Detection System")
st.sidebar.markdown("""
**About Project:**
This app uses a machine learning model to detect potential fraudulent transactions. Upload data or try the demo to get predictions and insights.

**Features:**
- Real-time summary
- Data cleaning preview
- Model predictions
- Fraud heatmap
- Risk categorization
- Download prediction reports
""")

st.sidebar.markdown("---")
st.sidebar.subheader("📥 Sample Dataset")
st.sidebar.markdown(
    "[Download Sample Dataset (Kaggle)](https://www.kaggle.com/datasets/ealaxi/paysim1)"
)

uploaded_file = st.sidebar.file_uploader("📤 Upload CSV file", type=['csv'])

# ============ Helper: Clean Input ============
def preprocess(df):
    drop_cols = ['Transaction_ID', 'User_ID', 'Timestamp', 'Fraud_Label']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # Encode object features
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    return df.select_dtypes(include=[np.number])

# ============ Tabs ============
tabs = st.tabs([
    "📊 Real-time Summary Dashboard", 
    "🧹 Data Preprocessing Preview", 
    "🤖 Model Prediction", 
    "📈 Model Performance", 
    "🗺️ Location-Based Heatmap",
    "🕒 Real-Time Transaction Simulation"  
])


# ============ Read Data ============
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("Upload a dataset using sidebar to begin!")
    st.stop()

# ============ Summary Dashboard ============
with tabs[0]:
    st.subheader("📊 Real-time Summary Dashboard")
    st.dataframe(df.head())
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", len(df))
    col2.metric("Fraud Cases", df['Fraud_Label'].sum() if 'Fraud_Label' in df.columns else "N/A")
    col3.metric("Non-Fraud", (df['Fraud_Label'] == 0).sum() if 'Fraud_Label' in df.columns else "N/A")

    st.bar_chart(df['Amount'] if 'Amount' in df.columns else df.select_dtypes(include=np.number).iloc[:, 0])

# ============ Preprocessing Preview ============
with tabs[1]:
    st.subheader("🧹 Data Preprocessing Preview")
    st.markdown("Here's how your dataset looks after preprocessing for model prediction.")
    df_clean = preprocess(df)
    st.dataframe(df_clean.head())

# ============ Prediction Tab ============
with tabs[2]:
    st.subheader("🤖 Model Prediction")

    if st.button("Run Prediction"):
        try:
            preds = model.predict(df_clean)
            probs = model.predict_proba(df_clean)[:, 1]

            df_result = df.copy()
            df_result['Fraud_Prediction'] = preds
            df_result['Fraud_Probability'] = probs
            df_result['Risk_Level'] = pd.cut(
                probs,
                bins=[0, 0.4, 0.7, 1.0],
                labels=["Low", "Medium", "High"]
            )

            st.success("✅ Predictions completed!")
            st.dataframe(df_result.head())

            # Download
            buffer = BytesIO()
            df_result.to_csv(buffer, index=False)
            st.download_button("⬇️ Download Report", buffer.getvalue(), file_name="fraud_predictions.csv", mime='text/csv')

        except Exception as e:
            st.error(f"Prediction Error: {e}")

# ============ Model Performance ============
with tabs[3]:
    st.subheader("📈 Model Performance Metrics")

    if 'Fraud_Label' in df.columns:
        y_true = df['Fraud_Label']
        y_pred = model.predict(df_clean)

        cm = confusion_matrix(y_true, y_pred)
        cr = classification_report(y_true, y_pred, output_dict=True)

        st.write("**Confusion Matrix:**")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.write("**Classification Report:**")
        st.json(cr)

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, model.predict_proba(df_clean)[:, 1])
        roc_auc = auc(fpr, tpr)
        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax2.plot([0, 1], [0, 1], linestyle='--')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend()
        st.pyplot(fig2)
    else:
        st.info("Ground truth labels not available in this dataset.")

# ============ Location Heatmap ============
with tabs[4]:
    st.subheader("🗺️ Location-Based Fraud Heatmap")
    if 'Location' in df.columns and 'Fraud_Label' in df.columns:
        fraud_map = df[df['Fraud_Label'] == 1]['Location'].value_counts()
        fig3, ax3 = plt.subplots()
        sns.barplot(x=fraud_map.values, y=fraud_map.index, palette="Reds", ax=ax3)
        ax3.set_xlabel("Fraud Cases")
        ax3.set_ylabel("Location")
        st.pyplot(fig3)
    else:
        st.info("Location or Fraud_Label column not found.")
# ============ Real-Time Transaction Simulation ============ #
with tabs[5]:
    st.subheader("🕒 Real-Time Transaction Simulation")

    import time

    # Button to start simulation
    if st.button("▶️ Start Simulation"):
        simulated_data = df_clean.sample(50).reset_index(drop=True)
        metrics_placeholder = st.empty()
        table_placeholder = st.empty()

        total = 0
        fraud_count = 0

        for i, row in simulated_data.iterrows():
            total += 1
            input_df = row.to_frame().T
            pred = model.predict(input_df)[0]

            if pred == 1:
                fraud_count += 1

            with metrics_placeholder.container():
                st.write(f"**Transaction #{i+1} Processed**")
                st.metric("Total Processed", total)
                st.metric("🚨 Detected Fraud", fraud_count)
                st.metric("✅ Legitimate", total - fraud_count)

            with table_placeholder.container():
                st.dataframe(input_df)

            time.sleep(1.0)  # delay for real-time effect
