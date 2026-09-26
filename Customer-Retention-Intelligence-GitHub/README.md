# Customer Retention Intelligence

An end-to-end telecom customer churn and retention intelligence system combining machine learning, deep learning, explainable AI, semantic retrieval, and an LLM-powered retention assistant.

## Project Overview

The project analyzes telecom customer data to identify customers with elevated churn risk and provide decision-support information for retention analysis.

## Key Components

- Exploratory data analysis
- Customer churn modeling
- Deep learning models using TensorFlow/Keras
- Advanced mixed-input neural network
- Class imbalance handling
- Model evaluation
- SHAP explainability
- Customer-level churn risk scoring
- Risk segmentation
- Knowledge base
- Semantic search using embeddings
- Retrieval-Augmented Generation (RAG)
- Groq-powered LLM retention assistant
- Streamlit dashboard

## Architecture

Customer Data
→ Data Preparation
→ Churn Models
→ Advanced Deep Learning Model
→ SHAP Explainability
→ Customer Risk Engine
→ Knowledge Base
→ Semantic Retrieval
→ LLM Retention Assistant
→ Streamlit Dashboard

## Risk Levels

The project uses the following project-defined risk bands:

- Low: below 30%
- Medium: 30% to below 60%
- High: 60% or above

These thresholds are project-defined and are not universal industry standards.

## Dashboard

The Streamlit application provides:

### Overview
- Customer population
- Risk distribution
- Risk by contract

### Customer Risk
- Customer-level risk score
- Predicted churn probability
- Contract information
- Tenure
- Internet service
- Payment method
- Monthly charges

### AI Retention Assistant
The assistant combines customer risk information with project knowledge retrieved through semantic search and generates natural-language decision-support responses using an LLM.

## Explainability

SHAP is used to explain model predictions.

SHAP explanations describe how model features contribute to an individual prediction. They should not be interpreted as proof that a feature causes churn.

## Limitations

- Churn probabilities are model predictions, not guaranteed outcomes.
- Risk bands are project-defined.
- SHAP describes model behavior rather than causal relationships.
- Retention recommendations are decision-support suggestions.
- The project does not contain controlled intervention data demonstrating that a particular retention action causes a specific outcome.

## Technology Stack

Python  
Pandas  
NumPy  
Scikit-learn  
TensorFlow / Keras  
SHAP  
Sentence Transformers  
Groq  
Streamlit  
Plotly

## Dataset

The project uses the telecom customer churn dataset containing customer demographics, services, account information, billing information, and churn status.

## Project Structure

```text
Customer-Retention-Intelligence/
├── data/
├── notebooks/
├── src/
├── app/
├── knowledge_base/
├── models/
├── reports/
├── rag/
├── tests/
├── train.py
├── evaluate.py
├── requirements.txt
├── .gitignore
└── README.md