# 🏠 Housing Price Prediction — TensorFlow Regression

A deep learning regression model that predicts house sale prices using the King County, USA dataset. Built with TensorFlow/Keras and scikit-learn.

---

## Overview

This project walks through the full machine learning pipeline — from exploratory data analysis to a trained neural network — to predict residential property prices based on structural, geographic, and temporal features.

---

## Dataset

[King County House Sales](https://www.kaggle.com/harlfoxem/housesalesprediction) — 21,613 home sales in King County, Washington (2014–2015).

| Feature | Description |
|---|---|
| `price` | Sale price (target) |
| `sqft_living` | Interior square footage |
| `grade` | Construction and design quality |
| `lat` / `long` | Geographic coordinates |
| `yr_built` | Year of construction |
| `yr_renovated` | Year of last renovation |
| `bedrooms`, `bathrooms` | Room counts |
| + 12 more features | |

---

## Pipeline

**EDA**
- Checked for missing values and examined feature distributions
- Identified `sqft_living`, `grade`, `sqft_above`, and `bathrooms` as the most correlated features with price
- Visualized geographic price hotspots using lat/long scatter plots
- Stripped top 1% price outliers for cleaner visual exploration

**Feature Engineering**
- Extracted `sale_year` and `sale_month` from the date column
- Applied **cyclical encoding** (`sin`/`cos`) to `sale_month` to preserve its circular nature
- Dropped `id`, `date`, `zipcode`, and `sale_month` (replaced by cyclical features)

**Preprocessing**
- 80/20 train-test split
- `StandardScaler` fit on training data only, applied to both sets

**Model Architecture**
```
Input → Dense(64, ReLU) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(32, ReLU) → Dense(1)
Optimizer: Adam | Loss: MSE | Callbacks: EarlyStopping(patience=40)
```

**Evaluation**
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- MAE as % of average price
- Actual vs predicted scatter plot
- Residual distribution and residuals vs predicted plots

---

## Results

| Metric | Value |
|---|---|
| MAE | ~$102,000 |
| MAE / Avg Price | ~19% |
| Model fit | Strong for homes under $2M, weaker for luxury properties |

---

## Requirements

- Python 3.10
- TensorFlow / Keras
- scikit-learn
- pandas, NumPy
- Matplotlib, Seaborn

---

## Run Locally

```bash
git clone https://github.com/MomoSalter/king_county_housing_price_predictor
cd king_county_housing_price_predictor
jupyter notebook housing_price_prediction.ipynb
```

Download `kc_house_data.csv` from [Kaggle](https://www.kaggle.com/harlfoxem/housesalesprediction) and place it in the project root.

---

## Project Structure

```
king_county_housing_price_predictor/
│
├── housing_price_prediction.ipynb   # Main notebook
├── kc_house_data.csv                # Dataset (download from Kaggle)
└── README.md
```

---

## Author

**Moaaz Ahmed**
- GitHub: [@Moaaz Ahmed](https://github.com/MoaazSalter)