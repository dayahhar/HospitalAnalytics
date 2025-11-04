import os
import streamlit as st
import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def explain_feature_importances(model, X, top_n=10):
    """Return a dataframe of top_n most important features (handles column mismatch)."""
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame({'feature': X.columns, 'importance': [np.nan]*len(X.columns)})

    n_features_model = len(model.feature_importances_)
    n_features_data = X.shape[1]

    # Align lengths safely
    if n_features_model != n_features_data:
        st.warning(f"⚠️ Feature mismatch: model has {n_features_model}, data has {n_features_data}. Aligning automatically.")
        min_len = min(n_features_model, n_features_data)
        fi = pd.DataFrame({
            'feature': X.columns[:min_len],
            'importance': model.feature_importances_[:min_len]
        })
    else:
        fi = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        })

    return fi.sort_values('importance', ascending=False).head(top_n)


try:
    from sklearn.metrics import root_mean_squared_error as _rmselib
    def rmse(y_true, y_pred):
        return float(_rmselib(y_true, y_pred))
except Exception:
    def rmse(y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

MODEL_DIR = 'models'
REFUSED_MODEL_PATH = os.path.join(MODEL_DIR, 'model_refused.pkl')
BEDS_MODEL_PATH = os.path.join(MODEL_DIR, 'available_beds_model.pkl')

@st.cache_data
def load_services(path='data/services_weekly.csv'):
    return pd.read_csv(path)

def prepare_global_features(df):
    df = df.copy()

    for col in ['patients_request', 'patients_admitted', 'patients_refused','patient_satisfaction', 'staff_morale', 'available_beds']:
        if col not in df.columns:
            df[col] = np.nan

    if 'month' in df.columns:
        try:
            df['month'] = pd.to_numeric(df['month'], errors='coerce')
        except Exception:
            df['month'] = df['month']
    else:
        df['month'] = np.nan

    numeric_cols = ['patients_request', 'patients_admitted', 'patients_refused', 'patient_satisfaction', 'staff_morale', 'available_beds']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            df[c] = df[c].fillna(df[c].median())
    
    if 'event' in df.columns:
        df['event'] = df['event'].fillna('none').astype(str)
        event_dummies = pd.get_dummies(df['event'], prefix='event')
        event_dummies = event_dummies.reindex(sorted(event_dummies.columns), axis=1)
        df = pd.concat([df, event_dummies], axis=1)
    else:
        df['event_none'] = 1

    base_features = ['patients_request', 'patients_admitted', 'patient_satisfaction', 'staff_morale','month']
    event_cols = [c for c in df.columns if c.startswith('event_')]
    features_with_event = base_features + event_cols

    df_modelA = df.dropna(subset=['patients_refused']) if 'patients_refused' in df.columns else df.copy()
    X_A = df_modelA[features_with_event].copy()
    y_A = df_modelA['patients_refused'].astype(float)

    df_modelB = df.dropna(subset=['available_beds']) if 'available_beds' in df.columns else df.copy()
    features_B = ['patients_request', 'patients_admitted', 'patients_refused', 'patient_satisfaction', 'staff_morale', 'month']

    if 'patients_refused' not in df.columns:
        df['patients_refused'] = 0
        features_B = [f for f in features_B if f in df.columns]

    X_B = df_modelB[[c for c in features_B if c in df_modelB.columns] + event_cols].copy()
    y_B = df_modelB['available_beds'].astype(float)

    return X_A, y_A, X_B, y_B, features_with_event, features_B

def train_and_save_rf(X, y, path, n_estimators=200, random_state=42):
    """Train RandomForestRegressor and save model to disk."""
    if X.shape[0] < 5:
        raise ValueError("Not enough data to train the model.")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        'rmse': rmse(y_test, preds),
        'r2': float(r2_score(y_test, preds))
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, path)

    return model, metrics

def load_model_if_exists(path, X=None, y=None, retrain_func=train_and_save_rf):
    """
    Safely load a model from disk. If missing or empty/corrupted, retrain automatically (if X and y provided).
    """
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            return joblib.load(path)
        except Exception as e:
            st.warning(f"⚠️ Model at {path} is corrupted ({e}). Retraining...")
    else:
        st.info(f"🧠 No model found at {path}. Training a new one...")

    if X is not None and y is not None:
        try:
            model, metrics = retrain_func(X, y, path)
            st.success(f"✅ Model trained and saved! RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.2f}")
            return model
        except Exception as e:
            st.error(f"Training failed: {e}")
            return None
    else:
        st.error("No data provided to train a new model.")
        return None

def explain_feature_importance(model, feature_names, top_n=10):
    if model is None:
        return pd.DataFrame()
    try:
        fi = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
        return fi.head(top_n).reset_index().rename(columns={'index': 'feature', 0: 'importance'})
    except Exception:
        return pd.DataFrame()
    
def app():
    st.title('Demand Predictions for Hospital Resources')
    st.markdown('Two seperate RandomForest models (global hospital level).')

    st.info('Feature set: patients_request, patients_admitted, patient_satisfaction, staff_morale, month, event dummies.')

    df = load_services()
    st.write(f'Data rows: {df.shape[0]}, columns: {df.shape[1]}')

    X_A, y_A, X_B, y_B, features_A, features_B = prepare_global_features(df)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Model A: Predict Patients Got Refused Beds')
        modelA = load_model_if_exists(REFUSED_MODEL_PATH, X_A, y_A)
        if modelA is not None:
            st.success('Loaded existing model: patients_refused_model')
            st.write('Example features used:', features_A)
            fiA = explain_feature_importances(modelA, X_A)
            if not fiA.empty:
                st.table(fiA)
        else:
            st.warning('No saved model found for patients_refused.')

        if st.button('Train/Re-train Model A (patients_refused)'):
            try:
                modelA, metricsA = train_and_save_rf(X_A, y_A, REFUSED_MODEL_PATH)
                st.success(f"Model A trained. RMSE: {metricsA['rmse']:.2f}, R2: {metricsA['r2']:.2f}")
                fiA = explain_feature_importances(modelA, X_A)
                if not fiA.empty:
                    st.table(fiA)
            except Exception as e:
                st.error(f"Training failed: {e}")
        
        st.markdown('**Make a quick prediction (Model A)**')
        with st.form('predict_refused'):
            pr = st.number_input('patients_request', value=float(df['patients_request'].median() if 'patients_request' in df.columns else 10))
            pa = st.number_input('patients_admitted', value=float(df['patients_admitted'].median() if 'patients_admitted' in df.columns else 5))
            psat = st.number_input('patient_satisfaction', value=float(df['patient_satisfaction'].median() if 'patient_satisfaction' in df.columns else 4.0))
            morale = st.number_input('staff_morale', value=float(df['staff_morale'].median() if 'staff_morale' in df.columns else 3.5))
            month = st.number_input('month (numeric)', value=int(df['month'].median() if 'month' in df.columns else 1))
            submitted = st.form_submit_button('Predict patients_refused')

        if submitted:
            modelA = load_model_if_exists(REFUSED_MODEL_PATH, X_A, y_A)
            if modelA is None:
                st.error('Model A not available. Please train the model first.')
            else:
                input_df = pd.DataFrame([{
                    'patients_request': pr,
                    'patients_admitted': pa,
                    'patient_satisfaction': psat,
                    'staff_morale': morale,
                    'month': month
                }])

            if not any(c.startswith('event_') for c in X_A.columns):
                pass
            else:  
                for c in X_A.columns:
                    if c.startswith('event_') and c not in input_df.columns:
                        input_df[c] = 0

            input_df = input_df.reindex(columns=X_A.columns, fill_value=0)
            pred_refused = modelA.predict(input_df)[0]
            st.metric('Predicted patients_refused', f"{pred_refused:.1f}")

    with col2:
        st.subheader('Model B: Predict Available Beds')
        modelB = load_model_if_exists(BEDS_MODEL_PATH, X_B, y_B)
        if modelB is not None:
            st.success('Loaded existing model: available_beds_model')
            st.write('Example features used:', features_B)
            fiB = explain_feature_importances(modelB, X_B)
            if not fiB.empty:
                st.table(fiB)
        else:
            st.warning('No saved model found for available_beds.')

        if st.button('Train/Re-train Model B (available_beds)'):
            try:
                modelB, metricsB = train_and_save_rf(X_B, y_B, BEDS_MODEL_PATH)
                st.success(f"Model B trained. RMSE: {metricsB['rmse']:.2f}, R2: {metricsB['r2']:.2f}")
                fiB = explain_feature_importances(modelB, X_B)
                if not fiB.empty:
                    st.table(fiB)
            except Exception as e:
                st.error(f"Training failed: {e}")

        st.markdown('**Make a quick prediction (Model B)**')
        with st.form('predict_beds'):
            pr_b = st.number_input('patients_request', value=float(df['patients_request'].median() if 'patients_request' in df.columns else 10))
            pa_b = st.number_input('patients_admitted', value=float(df['patients_admitted'].median() if 'patients_admitted' in df.columns else 5))
            pref_b = st.number_input('patients_refused', value=float(df['patients_refused'].median() if 'patients_refused' in df.columns else 1))
            psat_b = st.number_input('patient_satisfaction', value=float(df['patient_satisfaction'].median() if 'patient_satisfaction' in df.columns else 4.0))
            morale_b = st.number_input('staff_morale', value=float(df['staff_morale'].median() if 'staff_morale' in df.columns else 3.5))
            month_b = st.number_input('month (numeric)', value=int(df['month'].median() if 'month' in df.columns else 1))
            submitted_b = st.form_submit_button('Predict available_beds')

        if submitted_b:
            modelB = load_model_if_exists(BEDS_MODEL_PATH, X_B, y_B)
            if modelB is None:
                st.error('Model B not available. Please train the model first.')
            else:
                input_df_b = pd.DataFrame([{
                    'patients_request': pr_b,
                    'patients_admitted': pa_b,
                    'patients_refused': pref_b,
                    'patient_satisfaction': psat_b,
                    'staff_morale': morale_b,
                    'month': month_b
                }])

                # Add missing event columns if any
                for c in X_B.columns:
                    if c.startswith('event_') and c not in input_df_b.columns:
                        input_df_b[c] = 0

                # Reindex to match training features
                input_df_b = input_df_b.reindex(columns=X_B.columns, fill_value=0)

                # Make prediction
                pred_beds = modelB.predict(input_df_b)[0]
                st.metric('Predicted available_beds', f"{pred_beds:.1f}")

    st.markdown('---')
    st.write('Developed by Nur Hidayah Abdul Rahman.')

app()