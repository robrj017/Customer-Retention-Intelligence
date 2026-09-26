# Customer Retention Intelligence FAQ
## What does the model predict?
The model predicts the probability that a customer will churn based on the 
customer information available in the dataset.
## Does a high risk score mean the customer will definitely churn?
No.
A high score means the model estimates a higher probability of churn. It is 
not a guarantee.
## What does a risk score of 80 mean?
It means the model assigned approcimately 80% churn probability to that 
customer.
## What is the difference between risk score and actual churn?
Risk score is a model prediction.
Actual churn is what happened historically and is represented by the target variable.
## What does SHAP explain?
SHAP explains how individual features influenced a model prediction.
Positive SHAP values push the prediction toward higher churn probability.
Negative SHAP values push it toward lower churn probability.
## Does SHAP prove that a feature causes churn?
No 
SHAP explains the behaviour of the model. It does not establish causation.
## Why are multiple evaluation metrics used?
Because accuracy alone can be missleanding when the classes are imbalanced.
The project therefore also examines precision, recall,F1 score, ROC-AUC and
PR-AUC.
## What is the advanced deep-learning model?
It is a neural netwrok designed for tabular customer data. It combines 
normalized numerical variables with learned embeddings for categorical variables.
## Why are embeddings used?
Embeddings allow the neural network to learn numerical representations of
categorical values instead of treating category labels as ordinary numeric 
measurements.
## What is a high-risk customer?
In this project, a customer with predicted churn probability of 60% or above is
classified as high risk.
## Can the model tell us which retention action will definitely work?
No 
The current dataset does not contain controlled intervention outcomes, so the
mode cannot establish the casual effect of a retention action
## What should an employee do with a high-risk customer?
Review the customer's profile, examine the model's main risk contributors,
and then decide what account or service review is appropriate.
## Does the chatbot replace the churn model?
No
The chatbot is an interface around the project. It can explain model results,
retrieve customer information anf answer questions using the project knowledge base.
## Does the chatbot have access to information outside this project?
The intended project version is grounded in the project's customer data,
model outputs and knowledge-base documents.
## Is the project trained on customer conversations?
No.
The original dataset is structured telecom customer data. The NLP/RAG layer
is a separate knowledge component added to support explanation and business
questions.