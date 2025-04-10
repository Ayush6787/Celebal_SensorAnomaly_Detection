# 🧠 The problem is to predict anomaly based on the readings captured by sensors. Detection Using XGBoost

A machine learning pipeline for fraud detection using **XGBoost**. This project includes data preprocessing, feature engineering, model training with hyperparameter tuning (Optuna), and generating predictions on test data.

---

## 🚀 Project Overview

The goal is to build a **robust fraud detection classifier** that can generalize well and achieve a **high F1-score (above 0.75)** on unseen data. The project includes:

- 🔍 Data Cleaning & Feature Engineering
- 📈 Model Training (XGBoost)
- 🎯 Hyperparameter Tuning with Optuna
- 🦪 Resampling with SMOTE to handle class imbalance
- 🦪 Test Set Inference
- 📀 CSV Submission for evaluation

---

## 📂 Dataset

- `train.parquet`: Contains labeled transaction data with features and target column.
- `test.parquet`: Contains unlabeled transaction data for prediction.
- Columns include `X1, X2, ..., X5`, and additional time-based features engineered like `year, month, day, dayofweek, weekend`.

---

## 🛠️ Tools & Libraries

- Python 3.11
- pandas, numpy
- XGBoost
- Optuna (for tuning)
- scikit-learn
- imbalanced-learn (for SMOTE)
- Keras (optional for deep learning baseline)

---

## ⚖️ Feature Engineering

Engineered additional features from the `Date` column:

```python
df['year'] = pd.to_datetime(df['Date']).dt.year
df['month'] = pd.to_datetime(df['Date']).dt.month
df['day'] = pd.to_datetime(df['Date']).dt.day
df['dayofweek'] = pd.to_datetime(df['Date']).dt.dayofweek
df['weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
```

---

## ⚙️ Model Pipeline

1. **Preprocessing:**
   - Feature extraction from datetime
   - Dropping unnecessary columns (like `Date`)
   - Scaling using `RobustScaler`

2. **Resampling:**
   - Applied SMOTE to balance the dataset

3. **Model Training:**
   - Trained `XGBClassifier`
   - Tuned with Optuna for optimal performance
   - Scored using `F1-Score` with 3-fold cross-validation

---

## 📊 Results

- ✅ Final F1-Score: `0.78` (example)
- ✅ Achieved on validation with resampled and tuned model
- ✅ Predictions generated on `test.parquet` and saved to CSV

---

## 📁 Output

- `xgb_predictions.csv`: contains final predictions on the test set.

```csv
ID,Prediction
0,0
1,0
2,1
...
```

---

## 📌 How to Run

1. Clone the repo:
```bash
git clone https://github.com/your-username/fraud-detection-xgb.git
cd fraud-detection-xgb
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run training & prediction:
```bash
python main.py  # or Jupyter Notebook if using .ipynb
```

---

## 🙌 Acknowledgements

- Optuna for hyperparameter tuning
- XGBoost for its scalable tree boosting algorithm
- SMOTE from `imbalanced-learn` for class balancing

---

## ✨ Author

**Ayush Argonda**  
🔗 [LinkedIn](https://www.linkedin.com/) | 💻 Machine Learning Enthusiast | 🧠 Building projects from scratch

