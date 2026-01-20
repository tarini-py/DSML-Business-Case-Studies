# 🎓 Graduate Admissions Prediction – Business Case Study (Linear Regression)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Tarini%20Prasad%20Samantray-blue?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mr-tps/)

---
# Jamboree Education - Linear Regression

This project builds a data-driven system to predict a student’s **probability of admission** to graduate programs using academic and profile features. The goal is not only accurate prediction, but also extracting **actionable business insights** for education consulting companies like *Jamboree Education*.

---

## 📌 Problem Statement

Education consultants need to:

- Estimate a student’s admission chances reliably  
- Identify the **most impactful profile factors**  
- Guide students on where to invest effort (GRE, CGPA, research, etc.)  
- Optimize counseling resources

This project solves all four using regression modeling and statistical analysis.

---

## 📊 Dataset Overview

Features used:

- GRE Score  
- TOEFL Score  
- University Rating  
- SOP Strength  
- LOR Strength  
- CGPA  
- Research Experience  
- Target: Chance of Admit (0–1)

---

## ⚙️ Methodology

- Data cleaning & EDA  
- Feature scaling  
- Linear Regression using `scikit-learn`  
- Model validation using:
  - R²  
  - Adjusted R²  
  - RMSE  
  - MAE  
- Assumption checks:
  - Multicollinearity (VIF)
  - Residual analysis
  - Heteroscedasticity tests

---

## 🚀 Model Performance

| Metric | Value |
|--------|-------|
| R² (Test) | ~0.819 |
| Adjusted R² | ~0.818 |
| RMSE | ~0.061 |
| MAE | ~0.043 |

✅ Strong predictive accuracy  
✅ Stable generalization (no overfitting)  
✅ Production-ready baseline model

---

## 🧠 Key Insights (Business Focused)

### Most Important Predictors

1. CGPA (strongest)
2. GRE Score
3. Research Experience
4. TOEFL Score
5. LOR Strength

### Low Impact

- SOP score  
- University rating

---

## 💡 Actionable Recommendations

**For Students**
- Focus on CGPA improvement
- Invest in GRE preparation
- Gain research/internship experience
- Secure strong recommendation letters

**For Consulting Companies**
- Reallocate resources from SOP editing to academics & test prep
- Use model as a counseling decision engine
- Provide personalized improvement roadmaps

---

## 📈 Business Impact

- Higher student admission success rate
- Improved counseling efficiency
- Personalized guidance at scale
- Strong competitive differentiation
- Better student satisfaction & retention

---

## 🔮 Future Improvements

- Add GRE section-wise scores
- Include work experience & internships
- University ranking tiers
- Program-level admission history
- Try non-linear models (Random Forest, XGBoost)

---

## 🛠 Tech Stack

- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- Statsmodels  
- Jupyter Notebook  

---