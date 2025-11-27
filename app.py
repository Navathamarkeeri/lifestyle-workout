import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Workout Calories Explorer", layout="wide")

st.title("Workout Calories Explorer")
st.markdown("Use this app to explore model results and predictions produced by the notebook.")

DATA_DIR = os.path.abspath(os.path.dirname(__file__))

# Helper to load CSVs if present
def load_csv(name):
    path = os.path.join(DATA_DIR, name)
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.error(f"Failed to read {name}: {e}")
            return None
    return None

# Sidebar: choose view
view = st.sidebar.selectbox("View", [
    "Model comparison",
    "Feature importance",
    "Workout type analysis",
    "Predictions output",
    "Saved model (predict)"
])

if view == "Model comparison":
    df = load_csv("model_comparison_results.csv")
    if df is None:
        st.info("`model_comparison_results.csv` not found. Run the notebook to generate it.")
    else:
        st.header("Model comparison")
        st.dataframe(df)
        st.bar_chart(df.set_index('Model')[['R² Score']])

elif view == "Feature importance":
    df = load_csv("feature_importance_xgboost.csv")
    if df is None:
        st.info("`feature_importance_xgboost.csv` not found. Run the notebook to generate it.")
    else:
        st.header("Top feature importances (XGBoost)")
        st.dataframe(df.head(50))
        fig, ax = plt.subplots(figsize=(8, 6))
        top = df.head(15).sort_values('Importance')
        ax.barh(top['Feature'], top['Importance'], color='teal')
        ax.set_xlabel('Importance')
        ax.set_title('Top 15 Feature Importance - XGBoost')
        st.pyplot(fig)

elif view == "Workout type analysis":
    df = load_csv("workout_type_analysis.csv")
    if df is None:
        st.info("`workout_type_analysis.csv` not found. Run the notebook to generate it.")
    else:
        st.header("Workout type performance")
        st.dataframe(df)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(df['Workout_Type'], df['R²'], color='skyblue', edgecolor='black')
        ax.set_ylim(0, 1)
        ax.set_ylabel('R²')
        ax.set_title('R² by Workout Type')
        plt.xticks(rotation=45)
        st.pyplot(fig)

elif view == "Predictions output":
    df = load_csv("predictions_output.csv")
    if df is None:
        st.info("`predictions_output.csv` not found. Run the notebook to generate it.")
    else:
        st.header("Predictions vs Actual")
        st.dataframe(df.head(200))
        st.subheader("Error summary")
        if 'Error_XGBoost' in df.columns:
            st.write(df[['Error_XGBoost', 'Error_RandomForest', 'Error_LightGBM']].describe().T)
        else:
            st.write(df.describe())

elif view == "Saved model (predict)":
    st.header("Use a saved model for prediction")
    # Look for common model filenames
    candidate_files = [
        "xgb_model.pkl",
        "rf_model.pkl",
        "lgb_model.pkl",
        "best_model.pkl",
    ]
    found = [f for f in candidate_files if os.path.exists(os.path.join(DATA_DIR, f))]
    if not found:
        st.info("No saved model found (xgb_model.pkl / rf_model.pkl / lgb_model.pkl). If you want prediction from the trained model, save the model in the notebook as a .pkl file named `xgb_model.pkl` or similar and re-run this app.")
    else:
        model_path = os.path.join(DATA_DIR, found[0])
        st.write(f"Loading model: {found[0]}")
        try:
            model = joblib.load(model_path)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            model = None

        if model is not None:
                st.subheader("Quick prediction (enter numeric features)")
                st.write("Only a minimal interactive predictor is provided. If you saved preprocessing artifacts (`scaler.pkl`, `feature_list.pkl`, `label_encoders.pkl`) alongside the model, the app will attempt to use them for consistent preprocessing.")
                # Provide a minimal set of numeric inputs commonly used in the notebook
                col1, col2, col3 = st.columns(3)
                with col1:
                    session_duration = st.number_input('Session Duration (hours)', value=1.0, step=0.1)
                    avg_bpm = st.number_input('Avg BPM', value=120)
                    max_bpm = st.number_input('Max BPM', value=180)
                with col2:
                    weight = st.number_input('Weight (kg)', value=70.0)
                    height = st.number_input('Height (m)', value=1.75, step=0.01)
                    age = st.number_input('Age', value=30)
                with col3:
                    carbs = st.number_input('Carbs (g)', value=50)
                    proteins = st.number_input('Proteins (g)', value=20)
                    fats = st.number_input('Fats (g)', value=10)

                # Try to load optional preprocessing artifacts
                scaler = None
                feature_list = None
                label_encoders = None
                try:
                    if os.path.exists(os.path.join(DATA_DIR, 'scaler.pkl')):
                        scaler = joblib.load(os.path.join(DATA_DIR, 'scaler.pkl'))
                    if os.path.exists(os.path.join(DATA_DIR, 'feature_list.pkl')):
                        feature_list = joblib.load(os.path.join(DATA_DIR, 'feature_list.pkl'))
                    if os.path.exists(os.path.join(DATA_DIR, 'label_encoders.pkl')):
                        label_encoders = joblib.load(os.path.join(DATA_DIR, 'label_encoders.pkl'))
                except Exception as e:
                    st.warning(f"Preprocessing artifacts could not be loaded: {e}")

                if st.button('Predict'):
                    # Build a small sample dict from provided inputs
                    sample_input = {
                        'Session_Duration (hours)': session_duration,
                        'Avg_BPM': avg_bpm,
                        'Max_BPM': max_bpm,
                        'Weight (kg)': weight,
                        'Height (m)': height,
                        'Age': age,
                        'Carbs': carbs,
                        'Proteins': proteins,
                        'Fats': fats
                    }

                    try:
                        # If we have the feature_list, construct a full-row matching training features
                        if feature_list is not None:
                            row = {f: 0 for f in feature_list}
                            # fill numeric subset from sample_input when names match
                            for k, v in sample_input.items():
                                if k in row:
                                    row[k] = v
                            sample_df = pd.DataFrame([row], columns=feature_list)

                            # Apply scaler if available
                            if scaler is not None:
                                try:
                                    sample_transformed = scaler.transform(sample_df)
                                    pred = model.predict(sample_transformed)
                                except Exception:
                                    # fallback to raw
                                    pred = model.predict(sample_df)
                            else:
                                pred = model.predict(sample_df)
                        else:
                            sample_df = pd.DataFrame([sample_input])
                            pred = model.predict(sample_df)

                        st.success(f"Predicted Calories Burned: {float(pred[0]):.2f}")
                    except Exception as e:
                        st.error(f"Model prediction failed: {e}")


st.sidebar.markdown("---")
st.sidebar.write("Data files found in project:")
for f in [
    'model_comparison_results.csv',
    'feature_importance_xgboost.csv',
    'workout_type_analysis.csv',
    'predictions_output.csv'
]:
    if os.path.exists(os.path.join(DATA_DIR, f)):
        st.sidebar.write(f"- {f}")

st.sidebar.caption("If you want the app to run predictions, save a trained model as a .pkl in the project folder.")
