import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    page_icon="📱",
    layout="wide"
)

MODEL_PATH = Path("churn_model.pkl")

NUMERIC_FEATURES = [
    "Account length",
    "Area code",
    "Number vmail messages",
    "Total day minutes",
    "Total day calls",
    "Total day charge",
    "Total eve minutes",
    "Total eve calls",
    "Total eve charge",
    "Total night minutes",
    "Total night calls",
    "Total night charge",
    "Total intl minutes",
    "Total intl calls",
    "Total intl charge",
    "Customer service calls",
]

CATEGORICAL_FEATURES = [
    "State",
    "International plan",
    "Voice mail plan",
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def convert_churn(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return int(value)

    text = str(value).strip().lower()

    if text in {"yes", "true", "1", "churn", "churned"}:
        return 1
    if text in {"no", "false", "0", "not churn", "not_churn"}:
        return 0

    try:
        number = float(text)
        if number in (0, 1):
            return int(number)
    except ValueError:
        pass

    return np.nan


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def train_model_from_upload(uploaded_file):
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    data = pd.read_csv(uploaded_file)
    data.columns = data.columns.str.strip()

    if "Churn" not in data.columns:
        raise ValueError("CSV must contain a 'Churn' column.")

    data["Churn"] = data["Churn"].apply(convert_churn)
    data = data.dropna(subset=["Churn"]).copy()
    data["Churn"] = data["Churn"].astype(int)

    missing = [c for c in ALL_FEATURES if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = data[ALL_FEATURES].copy()
    y = data["Churn"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42,
            class_weight="balanced", n_jobs=-1
        )
    }

    results = []
    fitted = {}

    for name, estimator in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator)
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1 Score": f1_score(y_test, pred, zero_division=0),
            "ROC AUC": roc_auc_score(y_test, prob)
        })
        fitted[name] = pipe

    results_df = pd.DataFrame(results).sort_values("ROC AUC", ascending=False)
    best_name = results_df.iloc[0]["Model"]
    best_model = fitted[best_name]

    return best_model, best_name, results_df, data


st.title("📱 Telecom Customer Churn Prediction")
st.caption("Machine Learning application for predicting whether a telecom customer is likely to churn.")

with st.sidebar:
    st.header("Model")
    model = load_model()

    if model is not None:
        st.success("Saved model loaded")
    else:
        st.warning("No churn_model.pkl found")
        st.info("Upload Telco_Customer.csv below to train a model in the app.")

    uploaded = st.file_uploader(
        "Upload Telco_Customer (1).csv",
        type=["csv"],
        help="Use the same Telco dataset used for model training."
    )

    if uploaded is not None:
        if st.button("Train model from CSV", use_container_width=True):
            try:
                with st.spinner("Training models..."):
                    model, best_name, metrics, train_data = train_model_from_upload(uploaded)
                st.session_state["runtime_model"] = model
                st.session_state["metrics"] = metrics
                st.session_state["best_name"] = best_name
                st.session_state["train_rows"] = len(train_data)
                st.success(f"Best model: {best_name}")
            except Exception as exc:
                st.error(f"Training error: {exc}")

runtime_model = st.session_state.get("runtime_model")
if runtime_model is not None:
    model = runtime_model

if model is None:
    st.info("Please upload the CSV and click 'Train model from CSV' to continue.")
    st.stop()

if "metrics" in st.session_state:
    st.subheader("Model Performance")
    st.dataframe(
        st.session_state["metrics"].style.format({
            "Accuracy": "{:.3f}",
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "F1 Score": "{:.3f}",
            "ROC AUC": "{:.3f}",
        }),
        use_container_width=True
    )

st.divider()
st.subheader("Customer Information")

states = [
    "AK","AL","AR","AZ","CA","CO","CT","DC","DE","FL","GA","HI","IA","ID",
    "IL","IN","KS","KY","LA","MA","MD","ME","MI","MN","MO","MS","MT","NC",
    "ND","NE","NH","NJ","NM","NV","NY","OH","OK","OR","PA","RI","SC","SD",
    "TN","TX","UT","VA","VT","WA","WI","WV","WY"
]

c1, c2, c3 = st.columns(3)

with c1:
    state = st.selectbox("State", states)
    account_length = st.number_input("Account length", min_value=1, max_value=500, value=100)
    area_code = st.number_input("Area code", min_value=100, max_value=999, value=415)
    international_plan = st.selectbox("International plan", ["No", "Yes"])
    voice_mail_plan = st.selectbox("Voice mail plan", ["No", "Yes"])

with c2:
    number_vmail_messages = st.number_input("Number vmail messages", min_value=0, max_value=200, value=8)
    total_day_minutes = st.number_input("Total day minutes", min_value=0.0, max_value=1000.0, value=180.0)
    total_day_calls = st.number_input("Total day calls", min_value=0, max_value=500, value=100)
    total_day_charge = st.number_input("Total day charge", min_value=0.0, max_value=200.0, value=30.0)
    total_eve_minutes = st.number_input("Total eve minutes", min_value=0.0, max_value=1000.0, value=200.0)
    total_eve_calls = st.number_input("Total eve calls", min_value=0, max_value=500, value=100)
    total_eve_charge = st.number_input("Total eve charge", min_value=0.0, max_value=200.0, value=17.0)

with c3:
    total_night_minutes = st.number_input("Total night minutes", min_value=0.0, max_value=1000.0, value=200.0)
    total_night_calls = st.number_input("Total night calls", min_value=0, max_value=500, value=100)
    total_night_charge = st.number_input("Total night charge", min_value=0.0, max_value=200.0, value=9.0)
    total_intl_minutes = st.number_input("Total intl minutes", min_value=0.0, max_value=100.0, value=10.0)
    total_intl_calls = st.number_input("Total intl calls", min_value=0, max_value=100, value=4)
    total_intl_charge = st.number_input("Total intl charge", min_value=0.0, max_value=50.0, value=2.8)
    customer_service_calls = st.number_input("Customer service calls", min_value=0, max_value=50, value=1)

input_df = pd.DataFrame([{
    "State": state,
    "Account length": account_length,
    "Area code": area_code,
    "International plan": international_plan,
    "Voice mail plan": voice_mail_plan,
    "Number vmail messages": number_vmail_messages,
    "Total day minutes": total_day_minutes,
    "Total day calls": total_day_calls,
    "Total day charge": total_day_charge,
    "Total eve minutes": total_eve_minutes,
    "Total eve calls": total_eve_calls,
    "Total eve charge": total_eve_charge,
    "Total night minutes": total_night_minutes,
    "Total night calls": total_night_calls,
    "Total night charge": total_night_charge,
    "Total intl minutes": total_intl_minutes,
    "Total intl calls": total_intl_calls,
    "Total intl charge": total_intl_charge,
    "Customer service calls": customer_service_calls
}])

if st.button("🔮 Predict Customer Churn", type="primary", use_container_width=True):
    try:
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0, 1])

        if prediction == 1:
            st.error("⚠️ High Churn Risk")
            st.metric("Churn Probability", f"{probability * 100:.2f}%")
            st.write("Recommendation: consider retention offers, plan review, or proactive customer support.")
        else:
            st.success("✅ Low Churn Risk")
            st.metric("Churn Probability", f"{probability * 100:.2f}%")
            st.write("Recommendation: continue normal customer engagement.")
    except Exception as exc:
        st.error(f"Prediction error: {exc}")


st.markdown("""
<style>

/* ==============================
   MAIN APP BACKGROUND
   ============================== */

.stApp {
    background: linear-gradient(
        135deg,
        #061826 0%,
        #0b2d42 45%,
        #083b56 100%
    );
}

/* Main content area */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* ==============================
   SIDEBAR
   ============================== */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #04121f,
        #082b40
    );
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: yellow;
}

/* ==============================
   HEADINGS
   ============================== */

h1, h2, h3 {
    color: white !important;
}

p, label, .stMarkdown {
    color: #e6f7ff;
}

/* ==============================
   INPUT BOXES
   ============================== */

.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    background-color: #ffffff !important;
    color: #111111 !important;
    border-radius: 8px;
}

/* ==============================
   BUTTON
   ============================== */

.stButton > button {
    width: 100%;
    border-radius: 10px;
    font-weight: bold;
    padding: 12px;
    border: none;
    background: linear-gradient(
        90deg,
        #00b4d8,
        #0077b6
    );
    color: white;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.03);
}

/* ==============================
   DATAFRAME
   ============================== */

[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* ==============================
   METRIC CARDS
   ============================== */

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 15px;
    border: 1px solid rgba(255,255,255,0.15);
}


  ## FILE UPLOADER
  

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.08);
    padding: 15px;
    border-radius: 15px
    color: #04121f;


}

</style>
""", unsafe_allow_html=True)


