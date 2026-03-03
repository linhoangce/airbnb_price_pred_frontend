import streamlit as st

st.set_page_config(page_title="About", page_icon="ℹ️")

st.title("ℹ️ About This Project")

st.markdown("""
## 🎯 Project Overview
This project predicts AirBnB nightly prices using machine learning on 51,000+ listings across 6 major US cities.

## 📊 Dataset
- **Size**: 51,000 listings
- **Cities**: LA, SF, NYC, Chicago, Boston, DC
- **Features**: 149 original features
- **Target**: Nightly price (USD)

## 🤖 Methodology
1. **Data Preprocessing**: Handled missing values, encoded categorical variables, created interaction features
2. **Feature Engineering**: 50+ engineered features including polynomials and city-specific interactions
3. **Feature Selection**: Compared F-regression, RFE, and combined approaches
4. **Model Training**: Tested OLS, XGBoost, Random Forest
5. **Evaluation**: Cross-validation with 5 folds, tested on 20% holdout set

## 📈 Results
- **Best Model**: XGBoost
- **Test RMSE**: $89.50
- **R² Score**: 0.68
- **Key Features**: Accommodates, bedrooms, neighborhood price, city

## 🛠️ Tech Stack
- **Backend**: FastAPI
- **Frontend**: Streamlit
- **ML**: XGBoost, scikit-learn
- **Visualization**: Plotly, Matplotlib

## 👨‍💻 Developer
[Your Name]
- GitHub: [your-github]
- LinkedIn: [your-linkedin]
- Email: [your-email]
""")
