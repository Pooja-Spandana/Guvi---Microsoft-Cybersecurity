# Guvi---Microsoft-Cybersecurity

- Project Idea (in simple terms)
You are helping Security Operations Centers (SOCs) automatically decide whether a cybersecurity alert is:

A real attack (True Positive)
A safe/harmless situation (Benign Positive)
Or a false alarm (False Positive)

🔵 Goal:
Build a machine learning model that looks at past security incidents and predicts the correct label (TP, BP, FP) for new incidents — saving analysts tons of time.

🧠 Simple Real-world Example
Imagine a security guard sees 100 alarms going off every day.

Some alarms mean real danger (robbery happening) → True Positive
Some alarms mean it's safe (someone forgot their ID card but they're an employee) → Benign Positive
Some alarms mean it was triggered by mistake (like a cat passing by) → False Positive

You are training a robot to tell the guard which alarm is real, safe, or false — using data from past alarms!

- Project Flow
START
  ↓
Understand Business Problem
  ↓
Load Data (train.csv, test.csv)
  ↓
Initial Data Inspection
  ↓
Data Cleaning
   - Drop junk columns
   - Handle missing values
   - Remove duplicates
   - Clean column names
  ↓
Feature Engineering
   - Extract Year, Month, Day, Hour from Timestamp
   - Create 'isWeekend' flag
   - Create 'alerts_per_incident' feature
   - Create 'alerts_per_org' feature
   - Create 'top_category_flag', 'top_entity_type_flag'
   - Create 'is_rare_alert' feature
  ↓
Preprocessing
   - Label Encoding categorical columns
   - Fill missing numerical values
  ↓
Train-Validation Split (80%-20%)
  ↓
Model Building
   - Random Forest, XGBoost (start simple)
  ↓
Model Evaluation
   - Accuracy
   - Precision
   - Recall
   - Macro F1-Score
  ↓
Hyperparameter Tuning (optional)
  ↓
Final Model
  ↓
Apply Preprocessing to Test Set
  ↓
Make Predictions on Test Data
  ↓
Prepare Submission / Deployment
  ↓
END


- Full Preprocessing Function (for Train & Test)

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df, is_train=True, label_encoders=None):
    """
    Preprocess the input dataframe:
    - Drop unwanted columns
    - Feature engineering
    - Handle missing values
    - Label encode categorical variables
    - (For train set) returns label_encoders to reuse on test set
    
    Args:
        df (DataFrame): input dataset
        is_train (bool): if True, will fit LabelEncoders; else will transform using provided encoders
        label_encoders (dict): fitted encoders for test data if is_train=False

    Returns:
        df (DataFrame): preprocessed dataframe
        label_encoders (dict): fitted LabelEncoders (only if is_train=True)
    """

    # Columns to drop
    columns_to_drop = [
        'Sha256', 'FolderPath', 'RegistryValueName', 'OSFamily',
        'State', 'City', 'OAuthApplicationId', 'ResourceIdName',
        'RegistryKey', 'AccountSid', 'AccountUpn', 'ApplicationId', 'Id'
    ]
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Clean column names
    df.columns = df.columns.str.strip()

    # Timestamp Feature Engineering
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df['Year'] = df['Timestamp'].dt.year
    df['Month'] = df['Timestamp'].dt.month
    df['Day'] = df['Timestamp'].dt.day
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['isWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

    # Alerts per incident
    if 'IncidentId' in df.columns:
        alerts_per_incident = df.groupby('IncidentId').size().rename('alerts_per_incident')
        df = df.merge(alerts_per_incident, left_on='IncidentId', right_index=True, how='left')

    # Alerts per organization
    if 'OrgId' in df.columns:
        alerts_per_org = df.groupby('OrgId').size().rename('alerts_per_org')
        df = df.merge(alerts_per_org, left_on='OrgId', right_index=True, how='left')

    # Top category flag
    if 'Category' in df.columns:
        top_categories = df['Category'].value_counts().head(5).index
        df['top_category_flag'] = df['Category'].apply(lambda x: 1 if x in top_categories else 0)

    # Top entity type flag
    if 'EntityType' in df.columns:
        top_entity_types = ['Ip', 'User', 'MailMessage']
        df['top_entity_type_flag'] = df['EntityType'].apply(lambda x: 1 if x in top_entity_types else 0)

    # Rare alert flag
    if 'AlertId' in df.columns:
        alert_counts = df['AlertId'].value_counts()
        rare_alerts = alert_counts[alert_counts == 1].index
        df['is_rare_alert'] = df['AlertId'].apply(lambda x: 1 if x in rare_alerts else 0)

    # Handling Missing Values
    categorical_cols = df.select_dtypes(include=['object']).columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

    df[categorical_cols] = df[categorical_cols].fillna('Unknown')
    df[numerical_cols] = df[numerical_cols].fillna(-1)

    # Label Encoding
    if is_train:
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        return df, label_encoders
    else:
        # For test data, reuse encoders
        for col in categorical_cols:
            if col in label_encoders:
                le = label_encoders[col]
                df[col] = le.transform(df[col])
        return df


- On Train Data (First Time)
train_data = pd.read_csv('train.csv')
train_data_processed, label_encoders = preprocess_data(train_data, is_train=True)

- On Test Data (Apply same transformations)
test_data = pd.read_csv('test.csv')
test_data_processed = preprocess_data(test_data, is_train=False, label_encoders=label_encoders)


https://github.com/kadarmeeran465/CyberShield-Cybersecurity-Incident-Classification/blob/main/Cybersecurity_Incident_Classification_Data_Preprocessing_Exploration.ipynb

https://github.com/pavankethavath/Microsoft-Classifying-Cybersecurity-Incidents-with-ML/blob/main/Data_preprocessing_and_EDA.ipynb

https://github.com/Sirpi-57/Microsoft-Cyber-Security-Incident-Grade-Classification/blob/main/Source%20Code/Data%20PreProcessing.ipynb


Kaggle notebook
https://www.kaggle.com/code/sanjanasharma1/microsoft-security-incident-prediction#Initial-Data-Inspection

https://docs.python.org/3/library/venv.html#creating-virtual-environments   

