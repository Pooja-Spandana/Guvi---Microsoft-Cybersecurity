# Microsoft Cybersecurity Incident Classification 

### This project is a machine learning pipeline for automatically classifying cybersecurity incidents into one of three categories:  
- **Benign Positive (BP)**  
- **False Positive (FP)**  
- **True Positive (TP)**  

Developed as part of the **Guvi - Microsoft Cybersecurity Certification**, this project leverages real-world large-scale security event data (~4M+ records) to automate incident triage and reduce manual analyst workload.

---

## Problem Statement

Security Operation Centers (SOCs) often deal with millions of alerts daily. The goal of this project is to classify incidents into their correct triage categories using machine learning techniques to:

- Minimize false positives
- Prioritize true threats
- Automate benign cases for quicker resolution

---
## Approach & Methodology

1. **Data Cleaning & Preprocessing**
2. **Exploratory Data Analysis**
3. **Feature Engineering**
4. **Model Building**
5. **Model Evaluation**
6. **Best Performing Model**
   - **Random Forest + RUS + Hyperparameter Tuning**
   - Achieved **Macro F1 Score: 0.9337** on validation set
   - Evaluated on full train and test datasets

---
## Model Performance

### Final Model (Random Forest) Performance on Test Set:

- **Accuracy**: `0.9447`
- **Macro F1 Score**: `0.94`
- **Confusion Matrix**:
    ```
    [[1556707   48736   25499]
     [  38038  797927   32932]
     [  25496   46233 1351127]]
    ```

- **Class-wise Insights**:
  - **BenignPositive**: High precision and recall (0.96)
  - **FalsePositive**: Slightly lower precision but strong recall (0.92)
  - **TruePositive**: Excellent balance between precision and recall (0.95)

---

## Tools & Libraries

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost
- Imbalanced-learn (`RandomUnderSampler`)
- Joblib 

---