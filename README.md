# 🏦 Credit Scoring Model using Machine Learning

![Credit Scoring Banner](https://img.shields.io/badge/Internship-CodeAlpha-blueviolet?style=flat-square)

> A comprehensive ML-based solution to predict customer creditworthiness using synthetic financial and behavioral data.

---

## 📌 Project Overview

This project was developed during my internship at **CodeAlpha**. It is a robust **Credit Scoring System** built using machine learning techniques to classify individuals as **creditworthy** or **not creditworthy** based on various financial, behavioral, and demographic features.

The system includes:
- Data generation (synthetic but realistic)
- Feature engineering
- Model training using multiple classifiers
- Hyperparameter tuning
- Evaluation metrics
- Visualization (ROC, comparison heatmaps, feature importance)
- Real-time prediction for new customer data

---

## 🚀 Features

✅ Generate 10,000+ samples of realistic credit data  
✅ Over **30 engineered features** derived from raw attributes  
✅ Models trained: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting  
✅ Automated model evaluation using AUC, Precision, Recall, F1  
✅ ROC Curve plotting and heatmap comparison  
✅ **Hyperparameter tuning** using GridSearchCV  
✅ **Real-time creditworthiness prediction**  
✅ Visual feature importance (Random Forest)  

---

## 🧠 Technologies Used

| Tool/Library | Purpose |
|--------------|---------|
| Python | Core programming |
| Pandas, NumPy | Data manipulation |
| Scikit-learn | ML models, preprocessing, evaluation |
| Seaborn, Matplotlib | Visualizations |
| GridSearchCV | Hyperparameter tuning |

---

## 📊 Sample Metrics Output

- **Accuracy**: ~0.90+
- **Precision**: ~0.88+
- **Recall**: ~0.91+
- **AUC-ROC**: ~0.94+ (after tuning)

---

## 📂 Project Structure

```
credit-scoring-model/
│
├── credit_scoring.py     # Main Python script with full pipeline
├── README.md            # This file
└── requirements.txt     # All dependencies
```

---

## ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Shreyansh260/credit-scoring-model.git
   cd credit-scoring-model
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python credit_scoring_model.py
   ```

---

## 🧪 How It Works

1. Generates 10,000 samples with financial and behavioral features
2. Performs feature engineering and encodes categorical data
3. Trains and compares multiple ML models
4. Tunes hyperparameters (especially for Random Forest)
5. Evaluates and visualizes results
6. Predicts creditworthiness of a new customer with confidence level

---

## 🔍 Example Output

```
Customer Creditworthiness: ✅ APPROVED
Probability of being creditworthy: 0.937
Confidence level: 0.937
```

---

## 📌 Internship Note

This project was completed as part of my internship at **CodeAlpha**, where I was assigned the task to build a **credit risk prediction system** using machine learning techniques. This helped me understand the end-to-end ML pipeline, from feature design to model deployment.

---

## 📈 Future Improvements

- Integrate with a Streamlit dashboard for user-friendly UI
- Use real-world credit datasets for benchmarking
- Include XGBoost and LightGBM models for advanced comparison
- Model explainability with SHAP values

---

## 📧 Contact

**Shriyansh Singh Rathore**  
📧 shreyanshsinghrathore7@gmail.com  
📱 +91-8619277114  
🎓 B.Tech (AI & Data Science), Poornima University  

---

## 🌟 Acknowledgements

Thanks to **CodeAlpha** for the opportunity to work on real-world machine learning projects during the internship.

---

## 📜 License

This project is released under the MIT License.
