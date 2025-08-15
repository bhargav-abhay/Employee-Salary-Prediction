# Employ-Salary-Prediction for EDUNET FOUNDATION
Employee Salary Prediction
This repository contains a machine learning project focused on predicting employee salaries. The predictions are based on various features such as job type, degree, major, industry, years of experience, and distance from a metropolis.

 # Table of Contents
Problem Statement

Solution Approach

Features

Data Preprocessing

Model Building

Model Evaluation

Results and Conclusion

Future Work

Usage/Instructions



# Problem Statement
The primary goal of this project is to develop a robust machine learning model that can accurately predict whether an employee's annual income is greater than $50K (>50K) or less than or equal to $50K (<=50K), given a set of relevant attributes.

# Solution Approach
The project utilizes various machine learning techniques for binary classification. The dataset used is adult 3.csv, which initially includes a large number of samples. The current model training has been performed on a processed version of this dataset.

# Features
The following features were used as input for the salary classification model:

age: Age of the individual.

workclass: The type of employer.

fnlwgt: Final weight.

educational-num: The number of years of education.

marital-status: Marital status of the individual.

occupation: The individual's occupation.

race: The individual's race.

gender: The individual's gender.

capital-gain: Capital gains.

capital-loss: Capital losses.

hours-per-week: Number of hours worked per week.

native-country: Country of origin.

The target variable is income (categorized as <=50K or >50K).

# Data Preprocessing
The following steps were performed to prepare the data for model training:

Missing Value Handling:

? values in the occupation column were replaced with 'others'.

? values in the workclass column were replaced with 'notlist'.

Irrelevant Data Removal (Dimensionality Reduction & Filtering):

Rows where workclass was 'Without-pay' or 'Never-worked' were removed.

Rows where education was '5th-6th', '1st-4th', or 'Preschool' were removed.

The education and relationship columns were dropped, likely due to redundancy or low predictive power after considering educational-num and marital-status.

# Outlier Treatment:

Outliers in the age column were handled by filtering the data to include ages between 17 and 75, inclusive. Box plots were used to visualize the age distribution before and after filtering.

Categorical to Numerical Conversion (Encoding):

Categorical features (workclass, marital-status, occupation, race, gender, native-country) were converted into numerical representations using LabelEncoder.

For the Streamlit app, OneHotEncoder was also used for education, workclass, occupation, and gender along with StandardScaler for numerical features.

# Feature Scaling:

Numerical features were scaled to a range between 0 and 1 using MinMaxScaler. This helps in normalizing the input features for better model performance. StandardScaler was also used in the final pipeline for robust evaluation and deployment.

# Model Building
The dataset was split into training and testing sets (80% training, 20% testing) with stratification to maintain class proportions. Several machine learning classification models were trained and evaluated:

K-Nearest Neighbors (KNN)

# Logistic Regression

Multilayer Perceptron (MLPClassifier)

Random Forest Classifier

Support Vector Machine (SVM)

Gradient Boosting Classifier

# Model Evaluation
Models were evaluated based on accuracy_score and a detailed classification_report (including precision, recall, and F1-score) on the test set.

Here's a summary of the model accuracies observed:

Model	Accuracy (with StandardScaler)
Logistic Regression	0.8271
Random Forest	0.8654
KNN	0.8361
SVM	0.8541
Gradient Boosting	0.8707

Export to Sheets
The Gradient Boosting Classifier demonstrated the highest accuracy of 0.8707, making it the best-performing model for this classification task.

# Results and Conclusion
The Gradient Boosting model achieved robust performance in classifying employee incomes. The detailed classification report provides insights into its precision and recall for both income classes (<=50K and >50K).

# Future Work
Fine-tune the hyperparameters of the best-performing model (Gradient Boosting) further using more advanced techniques (e.g., GridSearchCV with a wider range of parameters).

Explore other advanced ensemble methods or deep learning architectures.

Consider feature engineering to create new, more informative features from the existing ones.

Investigate and mitigate potential biases in the dataset, especially regarding sensitive attributes like race and gender, to ensure fair predictions.

Usage/Instructions
This project can be used to predict the income class of new employees based on their attributes.



Bash

git clone https://github.com/bhargav-abhay/Employee-Salary-Prediction.git
Navigate to the project directory:

Bash

cd Employee-Salary-Prediction
Install required libraries:
Ensure you have the necessary libraries. You can create a requirements.txt if not already present, by running pip freeze > requirements.txt after installing all used libraries in your environment.

Bash

pip install pandas numpy matplotlib scikit-learn streamlit joblib
Place the dataset:
Ensure your dataset (adult 3.csv) is located at C:\Users\ABHAY TRIPATHI\OneDrive\Desktop\ibm project\adult 3.csv as referenced in the code, or update the path in the script.

# Run the Jupyter Notebook:
Execute the cells in your emp.ipynb notebook to preprocess data, train models, and save artifacts (best_model.pkl, encoder.pkl, scaler.pkl).

Run the Streamlit Application:
Once the model artifacts are saved, you can run the Streamlit app:

Bash

streamlit run app.py
This will launch the interactive web application in your browser, allowing you to input employee details and get salary class predictions.
