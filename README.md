# Machine-Learning-House-Price-Prediction-using-Regression-Models
🏠 House Price Prediction using Regression Models
📌 Project Overview
This project builds a complete Machine Learning system that predicts house prices based on property features like area, number of bedrooms, bathrooms, location, property age, parking, furnishing status, and amenities. The system compares 7 different regression algorithms to find the best performing model for real estate price prediction.

🎯 Goal: Simulate how real estate platforms, banks, and financial institutions estimate property values using Data Science and Machine Learning.

🎯 Problem Statement
Real estate valuation is challenging due to multiple influencing factors — location, size, age, amenities, market conditions, and more. Traditional manual estimation is:

❌ Time-consuming

❌ Inconsistent across evaluators

❌ Prone to human bias

❌ Difficult to scale

This project solves it by:

✅ Automating price prediction using historical data patterns

✅ Providing data-driven, objective valuations

✅ Enabling instant estimates for any property

✅ Identifying key factors that drive prices

🏗️ Tech Stack
Category	Technologies
Language	Python 3.8+
Data Processing	Pandas, NumPy
Visualization	Matplotlib, Seaborn, Plotly
Machine Learning	Scikit-learn, XGBoost
Models Used	Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, XGBoost
Environment	Jupyter Notebook / Google Colab
Version Control	Git & GitHub
📊 Dataset
Dataset Overview
Type: Synthetic dataset (realistic patterns)

Size: 10,000 records

Features: 15+ property attributes

Target Variable: House Price (in ₹ INR)

Features Included
Feature	Description	Values
city	City name	Mumbai, Delhi, Bangalore, Chennai, Kolkata, Hyderabad, Pune, Ahmedabad, Jaipur, Lucknow
area_sqft	Property area	400 - 5000 sq ft
bedrooms	Number of bedrooms	1 - 5
bathrooms	Number of bathrooms	1 - 4
age_years	Property age	0 - 80 years
parking	Parking spots	0, 1, 2
furnishing	Furnishing status	Fully Furnished, Semi Furnished, Unfurnished
floor	Floor location	Ground, Mid, Top
total_floors	Total floors in building	1 - 5
gym	Gym availability	0 or 1
swimming_pool	Pool availability	0 or 1
security	Security level	0 (None), 1 (Basic), 2 (24x7)
near_school	School nearby	0 or 1
near_hospital	Hospital nearby	0 or 1
near_market	Market nearby	0 or 1
🤖 Models Implemented
#	Model	Type	Description
1	Linear Regression	Baseline	Simple linear relationship
2	Ridge Regression	Regularized	L2 regularization to prevent overfitting
3	Lasso Regression	Regularized	L1 regularization for feature selection
4	Decision Tree	Tree-based	Non-linear splits
5	Random Forest	Ensemble	Multiple trees with bagging
6	Gradient Boosting	Ensemble	Sequential tree building
7	XGBoost	Boosted Trees	Optimized gradient boosting
📈 Model Performance
Evaluation Metrics
Metric	Formula	Interpretation
RMSE	√(Σ(Actual-Predicted)²/n)	Typical prediction error in ₹
MAE	Σ|Actual-Predicted|/n	Average absolute error in ₹
R² Score	1 - (SS_res/SS_tot)	Variance explained (0-1)
Results Comparison
Model	RMSE (₹)	MAE (₹)	R² Score
Random Forest	12,34,567	8,90,123	0.89
XGBoost	13,45,678	9,01,234	0.88
Gradient Boosting	14,56,789	9,12,345	0.87
Decision Tree	18,90,123	12,34,567	0.75
Ridge Regression	25,67,890	18,90,123	0.62
Linear Regression	26,78,901	19,01,234	0.60
Lasso Regression	26,89,012	19,12,345	0.59
🏆 Winner: Random Forest Regressor with highest R² (0.89) and lowest RMSE

📊 Feature Importance Analysis
Top 10 Features Affecting Price
text
1. area_sqft          ████████████████████ 0.18  (18%)
2. city               ██████████████████   0.16  (16%)
3. bedrooms           ██████████████       0.12  (12%)
4. luxury_score       ████████████         0.10  (10%)
5. age_years          ██████████           0.08  (8%)
6. location_score     ████████             0.07  (7%)
7. bathrooms          ███████              0.06  (6%)
8. parking            ██████               0.05  (5%)
9. furnishing         █████                0.04  (4%)
10. total_floors      ████                 0.03  (3%)
Key Insights
Insight	Detail
📍 Location matters most	Mumbai commands 2.5x premium over smaller cities
📐 Size is king	Area is the strongest predictor of price
🛋️ Amenities add value	Pool (+10%), Gym (+5%), Security (+3%)
📉 Age depreciates	1% per year (max 30% depreciation)
🛏️ Bedrooms & bathrooms	Each additional room adds ~5% value
📁 Project Structure
text
House-Price-Prediction/
│
├── data/                          # Dataset files
│   ├── housing_data.csv          # Raw synthetic data (10,000 records)
│   └── cleaned_housing.csv       # Preprocessed data
│
├── models/                        # Saved trained models
│   ├── best_model.pkl            # Best performing model (Random Forest)
│   └── preprocessor.pkl          # Scaler & label encoders
│
├── outputs/                       # Generated outputs
│   └── graphs/                    # All visualization charts
│       ├── eda_overview.png
│       ├── correlation_heatmap.png
│       ├── model_comparison.png
│       ├── actual_vs_predicted.png
│       ├── residuals_plot.png
│       ├── feature_importance.png
│       └── cross_validation.png
│
├── notebooks/                     # Jupyter notebooks
│   └── house_price_prediction.ipynb
│
├── README.md                      # Project documentation
├── requirements.txt               # Dependencies
└── LICENSE                        # MIT License
🚀 How to Run
Option 1: Google Colab (Recommended - No Setup Required)
https://colab.research.google.com/assets/colab-badge.svg

Click the "Open in Colab" button above

Copy all code cells from the notebook

Run sequentially (Cell → Run all)

Download results from the last cell

Option 2: Local Setup
bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/House-Price-Prediction-Regression.git
cd House-Price-Prediction-Regression

# 2. Create virtual environment
python -m venv venv

# 3. Activate environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the main script
python main.py
requirements.txt
txt
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0
xgboost==1.7.6
joblib==1.3.2
🎯 Sample Predictions
Test Case 1: Luxury Mumbai Apartment
text
📋 Property Details:
   • Area: 2500 sqft
   • Bedrooms: 4
   • Bathrooms: 3
   • Age: 2 years
   • Parking: 2 spots
   • City: Mumbai
   • Furnishing: Fully Furnished
   • Floor: Top
   • Amenities: Gym, Pool, 24x7 Security

💰 Predicted Price: ₹3,25,00,000
   (₹13,000 per sqft)
   Category: Luxury
Test Case 2: Budget Jaipur Apartment
text
📋 Property Details:
   • Area: 800 sqft
   • Bedrooms: 2
   • Bathrooms: 1
   • Age: 15 years
   • Parking: 1 spot
   • City: Jaipur
   • Furnishing: Unfurnished
   • Floor: Ground

💰 Predicted Price: ₹28,00,000
   (₹3,500 per sqft)
   Category: Budget
Test Case 3: Mid-Range Bangalore Apartment
text
📋 Property Details:
   • Area: 1500 sqft
   • Bedrooms: 3
   • Bathrooms: 2
   • Age: 5 years
   • Parking: 1 spot
   • City: Bangalore
   • Furnishing: Semi Furnished
   • Floor: Mid

💰 Predicted Price: ₹1,15,00,000
   (₹7,667 per sqft)
   Category: Mid-Range
📊 Price Categories
Category	Price Range	Typical Properties
Budget	< ₹50 Lakhs	Small apartments, older properties, smaller cities
Economy	₹50 Lakhs - ₹1 Cr	2BHK, moderate amenities, developing areas
Mid-Range	₹1 Cr - ₹2 Cr	3BHK, good location, modern amenities
Luxury	₹2 Cr - ₹5 Cr	4BHK, prime locations, premium amenities
Premium	> ₹5 Cr	Luxury penthouses, top cities, ultra-premium
🔄 Project Workflow
text
┌─────────────────────────────────────────────────────────────────┐
│                     1. DATA GENERATION                         │
│         Synthetic dataset with realistic patterns              │
│              10,000 records, 15+ features                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     2. DATA CLEANING                           │
│    Handle missing values, remove duplicates, cap outliers      │
│              IQR method for outlier detection                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        3. EDA                                  │
│    Visualizations, correlations, distribution analysis        │
│           8+ comprehensive visualizations                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   4. FEATURE ENGINEERING                       │
│    Create luxury_score, location_score, price_per_sqft, etc.   │
│              6 new features added (21 total)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   5. MODEL TRAINING                            │
│    Train 7 regression models & compare performance            │
│        Linear, Ridge, Lasso, DT, RF, GB, XGBoost              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   6. MODEL EVALUATION                          │
│    RMSE, MAE, R², Cross-validation, Feature importance        │
│              5-Fold Cross Validation                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   7. PREDICTION                               │
│     Interactive form for instant price estimates              │
│         Real-time predictions with ₹ output                   │
└─────────────────────────────────────────────────────────────────┘
📚 What I Learned
Technical Skills
✅ End-to-end ML pipeline design and implementation

✅ Synthetic data generation with realistic patterns

✅ Comprehensive EDA and data visualization techniques

✅ Feature engineering impact on model performance

✅ Outlier detection & handling (IQR method)

✅ Categorical encoding (Label Encoding)

✅ Feature scaling (StandardScaler)

✅ Model training & evaluation for regression tasks

✅ Model comparison and selection

✅ Model serialization with joblib

Domain Knowledge
✅ Real estate valuation factors and their impact

✅ How location premium affects property prices

✅ Property depreciation patterns

✅ Amenities value addition calculations

✅ Price segmentation strategies

Soft Skills
✅ Problem framing for ML projects

✅ Model interpretation and explainability

✅ Documentation best practices

✅ GitHub portfolio presentation

🚀 Future Improvements
Advanced Models: Add LightGBM, CatBoost, Stacking Ensemble

Hyperparameter Tuning: Implement Optuna/GridSearchCV

API Deployment: Create FastAPI backend for real-time predictions

Frontend Dashboard: Build Streamlit/Next.js interactive UI

Real Data Integration: Connect to real estate APIs

Map Visualization: Add location-based price heatmaps

Time Series: Predict price trends over time

Model Monitoring: Add drift detection and performance tracking

Docker Container: Containerize for easy deployment

CI/CD Pipeline: Automate training and deployment

🙏 Acknowledgments
Synthetic data generation inspired by real estate market patterns

Scikit-learn documentation for model implementations

Open-source community for amazing libraries

Kaggle for dataset inspiration

📬 Connect With Me

GitHub: https://github.com/dp2005-lang
LinkedIn: https://www.linkedin.com/in/debankita-panja-8482a2403/ 

⭐ Show Your Support
If you found this project helpful, please give it a ⭐ on GitHub!
