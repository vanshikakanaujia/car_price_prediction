Car Price Prediction using Machine Learning

OVERVIEW
This project aims to predict the selling price of used cars using Machine Learning techniques. The model is trained on historical car data and provides price estimations based on various features such as fuel type, transmission, kilometers driven, and car age.

PROBLEM STATEMENT
Determining the correct selling price of a used car is challenging due to multiple influencing factors. This project solves this problem by building a predictive model that estimates car prices accurately.

DATASET
The dataset contains details of different cars, including:

Car_Name
Year
Present_Price
Kms_Driven
Fuel_Type
Seller_Type
Transmission
Selling_Price (Target Variable)

TECHNOLOGIES USED
Python
Pandas
NumPy
Matplotlib
Scikit-learn

METHODOLOGY 
1. Data Collection and Loading
2. Data Preprocessing
Handling missing values
Encoding categorical variables
Feature engineering (Car Age)
3. Splitting dataset into training and testing sets
4. Model training using Linear Regression
5. Model evaluation using R² Score and MAE

MODEL USED:
1.Linear Regression
A supervised learning algorithm used for predicting continuous values. It establishes a relationship between input features and the target variable (car price).

RESULTS
-The model achieves good accuracy on both training and testing data
-Scatter plots show a strong correlation between actual and predicted values

PROJECT STRUCTURE
car-price-prediction/
│── car_data.csv
│── main.ipynb
│── README.md
│── requirements.txt

FUTURE IMPROVEMENTS 
Implement advanced models like Random Forest and XGBoost
Improve accuracy with feature selection
Deploy as a web application

ACKNOWLEDGEMENT
This project was developed as part of the Fundamentals of AI and ML course.
