
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sentence_transformers import SentenceTransformer
from groq import Groq
BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = BASE_DIR / "reports"
RAG_DIR = BASE_DIR / "rag"
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
customer_risk = load_customer_risk()
chunks_df = load_rag_chunks()
chunk_embeddings = load_embeddings()
embedding_model = load_embedding_model()
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
