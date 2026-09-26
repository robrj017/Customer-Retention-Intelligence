
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sentence_transformers import SentenceTransformer
from groq import Groq
from tensorflow.keras.models import load_model


BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = BASE_DIR / "reports"
RAG_DIR = BASE_DIR / "rag"
MODEL_DIR = BASE_DIR / "models"


st.set_page_config(
    page_title="Customer Retention Intelligence",
    page_icon="📊",
    layout="wide"
)


@st.cache_data
def load_customer_risk():
    return pd.read_csv(
        REPORT_DIR / "customer_risk_scores.csv"
    )




@st.cache_data
def load_rag_chunks():
    return pd.read_csv(
        RAG_DIR / "chunks.csv"
    )


@st.cache_data
def load_embeddings():
    return np.load(
        RAG_DIR / "chunk_embeddings.npy"
    )


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_churn_model():
    return load_model(
        MODEL_DIR / "advanced_churn_model.keras"
    )


customer_risk = load_customer_risk()
chunks_df = load_rag_chunks()
chunk_embeddings = load_embeddings()
embedding_model = load_embedding_model()
churn_model = load_churn_model()


numerical_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]


categorical_features = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]


def prepare_model_inputs(dataframe):
    inputs = {
        "numerical_features": (
            dataframe[numerical_features]
            .astype("float32")
            .values
        )
    }

    for column in categorical_features:
        inputs[column] = (
            dataframe[column]
            .astype(str)
            .values
        )

    return inputs


def get_customer_profile(customer_id):
    customer = customer_risk[
        customer_risk["customerID"].astype(str)
        == str(customer_id)
    ]

    if customer.empty:
        return None

    return customer.iloc[0].to_dict()


def get_customer_risk(customer_id):
    profile = get_customer_profile(customer_id)

    if profile is None:
        return None

    return {
        "customerID": profile["customerID"],
        "churn_probability": float(profile["RiskScore"]),
        "risk_score": float(profile["RiskScore"]),
        "risk_level": profile["RiskLevel"],
        "tenure": profile["tenure"],
        "contract": profile["Contract"],
        "internet_service": profile["InternetService"],
        "monthly_charges": float(profile["MonthlyCharges"]),
        "payment_method": profile["PaymentMethod"]
    }


def semantic_search(query, top_k=3):
    query_embedding = (
        embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]
    )

    similarities = chunk_embeddings @ query_embedding

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for index in top_indices:
        results.append({
            "document": chunks_df.iloc[index]["document"],
            "section": chunks_df.iloc[index]["section"],
            "score": float(similarities[index]),
            "text": chunks_df.iloc[index]["text"]
        })

    return results


def build_knowledge_context(question, top_k=3):
    results = semantic_search(
        question,
        top_k=top_k
    )

    context = []

    for result in results:
        context.append(
            f"Document: {result['document']}\n"
            f"Section: {result['section']}\n"
            f"Content: {result['text']}"
        )

    return "\n\n".join(context)


def get_llm_response(question, customer_id):
    api_key = st.secrets["GROQ_API_KEY"]

    client = Groq(
        api_key=api_key
    )

    customer_context = get_customer_risk(
        customer_id
    )

    if customer_context is None:
        return "Customer not found."

    customer_context = json.dumps(
        customer_context,
        indent=2
    )

    knowledge_context = build_knowledge_context(
        question,
        top_k=3
    )

    system_prompt = """
You are an AI retention assistant for a telecom
customer analytics project.
Use only the supplied customer information and
project knowledge.
Explain model predictions as predictions,
not guaranteed outcomes.
Do not claim that a feature causes churn unless
the supplied knowledge explicitly supports that
conclusion.
When discussing retention actions, present them
as decision-support suggestions.
If information is insufficient, clearly say so.
Keep responses concise, professional and useful
for a retention analyst.
"""

    user_prompt = f"""
Customer information:
{customer_context}

Relevant project knowledge:
{knowledge_context}

Analyst question:
{question}
"""
    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.2,
        max_tokens=500
    )
    return response.choices[0].message.content
st.title("Customer Retention Intelligence")
st.write(
    "AI-powered telecom churn risk analysis, "
    "customer intelligence and retention decision support."
)
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select a section",
    [
        "Overview",
        "Customer Risk",
        "Predict Customer Risk",
        "Retention Assistant"
    ]
)
if page == "Overview":
    st.header("Portfolio Overview")
    total_customers = len(customer_risk)
    high_risk = (
        customer_risk["RiskLevel"]
        == "High"
    ).sum()
    medium_risk = (
        customer_risk["RiskLevel"]
        == "Medium"
    ).sum()
    low_risk = (
        customer_risk["RiskLevel"]
        == "Low"
    ).sum()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Customers",
        f"{total_customers:,}"
    )
    col2.metric(
        "High Risk",
        f"{high_risk:,}"
    )
    col3.metric(
        "Medium Risk",
        f"{medium_risk:,}"
    )
    col4.metric(
        "Low Risk",
        f"{low_risk:,}"
    )
    risk_counts = (
        customer_risk["RiskLevel"]
        .value_counts()
        .reset_index()
    )
    risk_counts.columns = [
        "RiskLevel",
        "Customers"
    ]
    fig = px.bar(
        risk_counts,
        x="RiskLevel",
        y="Customers",
        title="Customer Risk Distribution",
        text="Customers"
    )
    st.plotly_chart(
        fig,
        use_container_width=True
    )
    st.subheader("Risk by Contract")
    contract_risk = (
        customer_risk
        .groupby(
            ["Contract", "RiskLevel"]
        )
        .size()
        .reset_index(
            name="Customers"
        )
    )
    fig_contract = px.bar(
        contract_risk,
        x="Contract",
        y="Customers",
        color="RiskLevel",
        barmode="group",
        title="Risk Distribution by Contract"
    )
    st.plotly_chart(
        fig_contract,
        use_container_width=True
    )
elif page == "Customer Risk":
    st.header("Customer Risk Explorer")
    customer_ids = (
        customer_risk["customerID"]
        .astype(str)
        .tolist()
    )
    customer_id = st.selectbox(
        "Select Customer",
        customer_ids
    )
    profile = get_customer_profile(
        customer_id
    )
    risk = get_customer_risk(
        customer_id
    )
    if profile is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Risk Score",
            f"{risk['risk_score']:.2f}%"
        )
        col2.metric(
            "Risk Level",
            risk["risk_level"]
        )
        col3.metric(
            "Tenure",
            f"{risk['tenure']} months"
        )
        st.subheader("Customer Profile")
        profile_data = {
            "Customer ID": profile["customerID"],
            "Contract": profile["Contract"],
            "Internet Service": profile["InternetService"],
            "Payment Method": profile["PaymentMethod"],
            "Monthly Charges": f"${float(profile['MonthlyCharges']):.2f}",
            "Tenure": f"{profile['tenure']} months"
        }
        st.table(
            pd.DataFrame(
                profile_data.items(),
                columns=["Attribute", "Value"]
            )
        )
        st.subheader("Risk Interpretation")
        st.info(
            f"The model assigns this customer a "
            f"predicted churn probability of "
            f"{risk['risk_score']:.2f}%."
        )
        st.caption(
            "Risk scores are model predictions and "
            "should be used as decision-support signals."
        )
elif page == "Predict Customer Risk":
    st.header("Predict Customer Risk")
    st.write(
        "Enter a customer profile to generate a "
        "churn probability and risk score using "
        "the trained deep-learning model."
    )
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            tenure = st.number_input(
                "Tenure (months)",
                min_value=0,
                max_value=100,
                value=12
            )
            monthly_charges = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                value=70.0,
                step=0.01
            )
            total_charges = st.number_input(
                "Total Charges",
                min_value=0.0,
                value=840.0,
                step=0.01
            )
            gender = st.selectbox(
                "Gender",
                ["Female", "Male"]
            )
            senior_citizen = st.selectbox(
                "Senior Citizen",
                [0, 1]
            )
            partner = st.selectbox(
                "Partner",
                ["No", "Yes"]
            )
            dependents = st.selectbox(
                "Dependents",
                ["No", "Yes"]
            )
        with col2:
            phone_service = st.selectbox(
                "Phone Service",
                ["No", "Yes"]
            )
            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No phone service", "No", "Yes"]
            )
            internet_service = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )
            online_security = st.selectbox(
                "Online Security",
                ["No internet service", "No", "Yes"]
            )
            online_backup = st.selectbox(
                "Online Backup",
                ["No internet service", "No", "Yes"]
            )
            device_protection = st.selectbox(
                "Device Protection",
                ["No internet service", "No", "Yes"]
            )
            tech_support = st.selectbox(
                "Tech Support",
                ["No internet service", "No", "Yes"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV",
                ["No internet service", "No", "Yes"]
            )
        with col3:
            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["No internet service", "No", "Yes"]
            )
            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )
            paperless_billing = st.selectbox(
                "Paperless Billing",
                ["No", "Yes"]
            )
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )
        predict_button = st.form_submit_button(
            "Predict Risk"
        )
    if predict_button:
        input_data = pd.DataFrame([{
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method
        }])
        model_inputs = prepare_model_inputs(
            input_data
        )
        prediction = churn_model.predict(
            model_inputs,
            verbose=0
        )[0][0]
        risk_score = float(
            prediction * 100
        )
        if risk_score < 30:
            risk_level = "Low"
        elif risk_score < 60:
            risk_level = "Medium"
        else:
            risk_level = "High"
        st.subheader("Prediction Result")
        col1, col2 = st.columns(2)
        col1.metric(
            "Predicted Churn Probability",
            f"{risk_score:.2f}%"
        )
        col2.metric(
            "Risk Level",
            risk_level
        )
        st.subheader("Customer Input")
        display_data = input_data.T.reset_index()
        display_data.columns = [
            "Feature",
            "Value"
        ]
        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )
        st.info(
            f"The trained model predicts a "
            f"{risk_score:.2f}% churn probability "
            f"for the supplied customer profile."
        )
        st.caption(
            "This prediction is model-generated decision "
            "support and does not guarantee that the "
            "customer will churn."
        )
elif page == "Retention Assistant":
    st.header("AI Retention Assistant")
    customer_ids = (
        customer_risk["customerID"]
        .astype(str)
        .tolist()
    )
    customer_id = st.selectbox(
        "Customer",
        customer_ids
    )
    question = st.text_area(
        "Ask a retention question",
        placeholder=(
            "What should we review for this customer?"
        )
    )
    if st.button("Analyze Customer"):
        if not question.strip():
            st.warning(
                "Please enter a question."
            )
        else:
            with st.spinner(
                "Analyzing customer..."
            ):
                answer = get_llm_response(
                    question,
                    customer_id
                )
            st.subheader(
                "Retention Analysis"
            )
            st.write(answer)
            st.caption(
                "The assistant uses model predictions "
                "and project knowledge as decision-support "
                "information. It does not establish causal "
                "relationships."
            )

