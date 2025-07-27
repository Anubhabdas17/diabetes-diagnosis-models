# 🧠 Diabetes Prediction using Machine Learning

This project predicts the likelihood of diabetes in patients using machine learning techniques on the Pima Indians Diabetes dataset. Two models are developed and compared: **K-Nearest Neighbors (KNN)** and **XGBoost**, with explainability added using **SHAP**.

---

## 📁 Dataset
- **Source**: [Kaggle - Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Features include**: Glucose, Blood Pressure, BMI, Insulin, Skin Thickness, Age, Pregnancies, and more.

---

## 📌 Project Workflow

1. **Data Cleaning**:
   - Handled missing or invalid (zero) values in medical fields.
   - Imputed missing values with mean/median strategies.

2. **EDA (Exploratory Data Analysis)**:
   - Visualized distributions and correlations.
   - Used heatmaps, scatter plots, and missing value maps.

3. **Preprocessing**:
   - Feature scaling with `StandardScaler`.
   - Train-test split (80/20).

4. **Modeling**:
   - ✅ **K-Nearest Neighbors (KNN)** with GridSearchCV for hyperparameter tuning.
   - ✅ **XGBoost** with parameter tuning and evaluation.

5. **Evaluation Metrics**:
   - Accuracy, Confusion Matrix, Classification Report
   - ROC Curve and AUC Score

6. **Model Explainability**:
   - Applied SHAP (SHapley Additive exPlanations) to explain XGBoost predictions.

---

## 🧪 Libraries Used

- `pandas`, `numpy`
- `matplotlib`, `seaborn`, `missingno`
- `scikit-learn`, `xgboost`, `mlxtend`
- `shap`

---

## 📈 Results

### 🔍 XGBoost

- **Accuracy:** 74%
- **ROC AUC Score:** 0.596
- **F1 Score (Class 1 - Diabetic):** 0.62

> XGBoost gave moderate overall accuracy but struggled with distinguishing between classes, particularly diabetic cases. SHAP was used to interpret its predictions.

---

### 🔍 K-Nearest Neighbors (KNN)

- **Accuracy:** 77%
- **ROC AUC Score:** 0.819
- **F1 Score (Class 1 - Diabetic):** 0.64

> KNN outperformed XGBoost in both accuracy and ROC AUC, showing better balance in prediction across both classes.

---

## Dataset
- Included in this repository as `diabetes.csv`
- Original source: [Kaggle - Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

---

## 🛠️ How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/diabetes-prediction-ml.git
   cd diabetes-prediction-ml
