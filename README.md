# Customer-Churn-Analysis-in-Banking-with-Artificial-Neural-Networks
This project aims to predict whether a customer will churn (leave the bank) or stay using an Artificial Neural Network (ANN). The dataset used contains various customer attributes and the target variable Exited, indicating whether the customer churned.
## dataset
The dataset contains the following columns:

- RowNumber (int64)
- CustomerId (int64)
- Surname (object)
- CreditScore (int64)
- Geography (object)
- Gender (object)
- Age (int64)
- Tenure (int64)
- Balance (float64)
- NumOfProducts (int64)
- HasCrCard (int64)
- IsActiveMember (int64)
- EstimatedSalary (float64)
- Exited (int64, target variable)
  ## Exploratory Data Analysis (EDA)
### Proportion of Customers Who Retained and Churned
A pie chart was created to visualize the proportion of customers who have retained and churned.
![image](my_pie.png)
## Categorical Variable Analysis
- A bar chart was plotted to analyze the categorical variables. Key findings include:

- Most customers are from France, while the majority of customers who churned are from Germany.
- The proportion of female customers who churned is higher than that of male customers.
- Customers with credit cards have a higher churn rate.
- Inactive members have a greater churn rate, and the overall proportion of inactive members is high.
![image](my_bar.png)
## Correlation Heatmap
A heatmap was plotted to visualize the correlation between various features and the target variable Exited.
![image](my_cor.png)
