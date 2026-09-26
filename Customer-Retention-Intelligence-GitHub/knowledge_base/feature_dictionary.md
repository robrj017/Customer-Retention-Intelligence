# Customer Churn Feature Dictionary
## Purpose
This document explains the main fields used in the customer churn project.
It is intended to help analysts and the retention assistant understand the 
customer data without having to inspect the raw dataset.
## Customer Information 
### customerID
Unique identifier assigned to each customer
### gender
Customer gender recorded in the dataset
### SeniorCitizen 
Indicates whether the customer belongs to the senior-citizen category
### Partner 
Indicates whether the customer has a partner.
### Dependents 
Indicates whether the customer has dependents.
## Service Information
### PhoneService
Indicates whether the customer has phone service.
### MultipleLines
Indicates whether the customer has multiple phone lines.
### InternetService
Internet service type used by the customer

Possible categories in the dataset include DSL,Fiber optic, and No.
### OnlineSecurity
Indicates whether the customer has an online security service
### OnlineBackup
Indicated whether the customer has an online-backup service.
### DeviceProtection 
Indicated whether the customer has device-protection service.
### TechSupport 
Indicates whether the customer has technical-support service.
### StreamingTV
Indicated whether the customer has streaming-TV service.
### StreamingMovies
Indicated whether the customer has streaming-movie service.
## Account Information 
### tenure
Number of months the customer has stayed with the company.
### Contract
Contract type associated with the customer.
### PaperlessBilling 
Indicates whether the customer uses paperless billing.
### PaymentMethod
Payment method used by the customer.
### MonthlyCharges 
Customer's monthly service charges.
### TotalCharges 
Total charges recorded for the customer.
## Target
### Churn 
Indicates whether the customer churned.
In the modeling pipeline:
- 0 = Stayed
- 1 = Churned
## Model Risk Fields
### ChurnProbability 
Probability produced by the advanced churn model.
### RiskScore 
Churn probability expressed on a 0-100 scale.
### RiskLevel
Project-defined risk band:
- Low: probability below 30%
- Medium: probability form 30% to below 60%
- High: probability of 60% or above
These thresholds are project-defined risk bands and are not universal industry standards.