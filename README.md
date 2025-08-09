<h1>**Machine Learning-Based Prediction of Employee Attrition**</h1>

---

Attrition is a silent business killer. Every employee resignation costs companies not just money, but operational continuity, morale, and hard-earned expertise. As part of Rakamin Data Science Bootcamp Batch 54, my team and I, "Asterisk Celestials", were tasked to tackle this problem for a fictional company, XYZ Co., with a glaring 16.2% annual attrition rate.
Our objective: Leverage data science to predict which employees are at risk of leaving, reduce attrition in critical roles by 15%, and cut wasted training costs by the same margin.

---

Project Overview
Business Problem:
High employee turnover leading to significant financial and operational losses.
Project Goals:
Predict employee attrition with high precision.
Provide actionable insights for HR to proactively intervene.
Quantify business impact in terms of cost savings.

---

The Data: A Holistic View of Employee Attributes
We worked with 4,410 employee records sourced from multiple datasets:
General Employee Data (age, salary, job role, etc.)
Employee Satisfaction Surveys
Managerial Performance Evaluations
Attendance Logs (In Time & Out Time Tables)

Target Variable:
 Attrition (Binary: Yes/No)

---

Data Preparation & Feature Engineering
Highlights:
Merged datasets using Employee IDs.
Extracted AvgWorkingHours by calculating daily differences between "In Time" and "Out Time" for each employee.
Imputed missing values (Median for numeric, Mode for categorical).
Addressed outliers using RobustScaler.
Categorical encoding: One-Hot Encoding for nominal features and Ordinal Encoding for ordered features.
Resolved class imbalance using SMOTENC, as the attrition class was significantly underrepresented.

Final Feature Set Included:
Age, TotalWorkingYears, NumCompaniesWorked, AvgWorkingHours, JobRole, Department, MaritalStatus, JobSatisfaction, WorkLifeBalance, and more.

---

Modeling & Experimentation
I developed and compared several models (and these are top 3):
Random Forest (Tuned):
Accuracy: 98.03%
F1-Score: 94.11%
ROC-AUC: 99.58%
XGBoost:
Accuracy: 97.58%
F1-Score: 92.00%
ROC-AUC: 99.21%
Decision Tree:
Accuracy: 93.66%
F1-Score: 79.41%
ROC-AUC: 89.80%

After rigorous hyperparameter tuning, the Random Forest model outperformed others in accuracy and robustness.

---

Explainability & Fairness Analysis
Using SHAP values, we derived key attrition drivers:
Employees working more than 7.5 hours per day were at higher risk.
Younger, less experienced employees exhibited greater attrition likelihood.
Single employees were more prone to leave compared to married counterparts.

Fairness checks were conducted to ensure no demographic groups were unfairly targeted, with mitigations applied for minor gaps observed in marital status and department categories.

---

Business Impact: Quantified Savings & Risk Reduction
Without the model: Estimated annual loss of $681,000.
With the model: Potential loss reduced to $196,500.
Total amount saved: $593,500.
Attrition risk in critical roles was reduced by up to 79.8%. Model-driven HR interventions improved precision in retaining key employees.

---

Deployment: Streamlit Monitoring App
To ensure the model's longevity and maintainability, I developed a Streamlit app for:
Monitoring model performance (Accuracy, F1, ROC-AUC).
Tracking prediction errors and data drift.
Enabling scheduled and retraining.

---

Challenges Faced
Hardware Constraints: Limited resources increased model training time.
Hyperparameter Search Space: Expanding grid search could yield even better results.
Data Limitations: Certain features were removed due to constant values or missing data.

---

Conclusion: Data Science Driving Real Business Outcomes
Through this project, I learned how to combine technical machine learning workflows with business impact analysis to create a data-driven HR solution. The model didn't just predict attrition - it offered actionable strategies to prevent it, directly impacting the company's bottom line.
This end-to-end project - from data preprocessing to deployment - reinforced my skills in feature engineering, model tuning, explainability, and business value translation.

---

What's Next?
Enhance hyperparameter optimization with more advanced search techniques.
Improve deployment UI/UX for broader HR adoption.

---

Project Links:
GitHub Repository: https://github.com/fairulrifqi/Asterisk-Celestials
Notebook: https://colab.research.google.com/drive/13kwQBO7HsPMWcHgYO6RwjSc2rIpbevNz#scrollTo=CcxprjMNK52k
App Demo: https://asterisk-celestials-ec2jshkd3jgnm9wgsfafhr.streamlit.app/
Full Documentation & Reports: https://drive.google.com/drive/folders/1nD4RkgzY56v0mcJwQZFFDJIGE6PIKEe4
Final Report: https://www.canva.com/design/DAGtZqTNnbo/J4eNjTSJrqjBV8NtYTjOcA/edit?utm_content=DAGtZqTNnbo&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

---

Contact
Mail: fairulrifqi962@gmail.com
LinkedIn: linkedin.com/in/fairulrifqi
