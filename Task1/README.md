*Titanic Survival Prediction - CODSOFT Internship Task 1* 

Project Overview
This project aims to build a Machine Learning model that predicts whether a passenger survived the Titanic disaster.

The model uses passenger information such as age, gender, passenger class, fare, and embarkation port.

Dataset Information
The dataset includes:

- PassengerId
- Pclass (Passenger Class)
- Name
- Sex
- Age
- SibSp
- Parch
- Ticket
- Fare
- Cabin
- Embarked
- Survived (Target Variable)

Data Preprocessing Steps

1. Filled missing values in:
   - Age (mean)
   - Fare (mean)
   - Embarked (mode)

2. Dropped unnecessary columns:
   - Name
   - Ticket
   - Cabin
   - PassengerId

3. Converted categorical data:
   - Sex → Numerical
   - Embarked → Numerical

Model Used

Random Forest Classifier

Model Performance

Achieved approximately 80–85% accuracy on test data.


How to Run
1. Install dependencies:

   pip install -r requirements.txt

2. Run the model:

   py titanic_model.py
