# 🏥 Health Insurance Cost Prediction

A machine learning project that analyzes and predicts health insurance costs based on personal attributes using various regression algorithms.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Key Findings](#key-findings)
- [Usage](#usage)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project explores factors that influence health insurance premiums and builds predictive models to estimate insurance costs. Through comprehensive data analysis and multiple machine learning approaches, we identify the most significant factors affecting insurance charges.

**Primary Goal**: Develop accurate models to predict annual health insurance costs based on demographic and health-related features.

## 📊 Dataset

The dataset contains **1,338 records** with the following features:

| Feature | Description | Type |
|---------|-------------|------|
| age | Age of the primary beneficiary | Numeric |
| sex | Gender (male/female) | Categorical |
| bmi | Body Mass Index | Numeric |
| children | Number of dependents covered | Numeric |
| smoker | Smoking status (yes/no) | Categorical |
| region | Residential area (northeast, southeast, southwest, northwest) | Categorical |
| charges | Annual medical costs (Target Variable) | Numeric |

**Note**: The dataset has no missing values, making it clean and ready for analysis.

## 🛠️ Technologies Used

- **Python 3.8+**
- **Libraries**:
  - `pandas` - Data manipulation and analysis
  - `numpy` - Numerical computing
  - `matplotlib` - Data visualization
  - `seaborn` - Statistical visualization
  - `scikit-learn` - Machine learning algorithms
  - `jupyter` - Interactive notebooks

## 💻 Installation

### Prerequisites
```bash
python >= 3.8
pip
```

### Setup

1. Clone the repository
```bash
git clone https://github.com/yourusername/insurance-cost-prediction.git
cd insurance-cost-prediction
```

2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages
```bash
pip install -r requirements.txt
```

4. Launch Jupyter Notebook
```bash
jupyter notebook
```

5. Open `insurance-cost-with-linear-regression.ipynb`

## 🔍 Exploratory Data Analysis

### Key Insights from EDA:

1. **Distribution of Charges**: Insurance costs show a right-skewed distribution with most people having lower charges
2. **Regional Analysis**: Southeast has the highest total charges, Southwest has the lowest
3. **Smoking Impact**: Smokers have significantly higher insurance costs across all demographics
4. **Age & BMI**: Positive correlation with charges, especially pronounced for smokers
5. **Family Size**: Weak correlation with costs; people with children tend to smoke less

### Visualizations Include:
- Distribution plots (original and log-transformed)
- Regional comparison charts
- Scatter plots with regression lines
- Correlation heatmap
- Violin plots for categorical comparisons
- Feature importance charts

## 🤖 Models Implemented

We implemented and compared five different regression algorithms:

| Model | Description | R² Score |
|-------|-------------|----------|
| **Polynomial Regression** | Linear regression with degree-2 polynomial features | **0.870** |
| **Random Forest** | Ensemble of 100 decision trees | **0.860** |
| **Linear Regression** | Basic linear relationship modeling | 0.751 |
| **Ridge Regression** | Linear regression with L2 regularization | 0.750 |
| **Lasso Regression** | Linear regression with L1 regularization | 0.749 |

## 📈 Results

### Best Model: Polynomial Regression
- **R² Score**: 0.870
- **Mean Absolute Error**: ~$2,500-3,000
- **RMSE**: ~$4,500-5,000

### Model Performance Comparison
![Model Comparison](images/model_comparison.png)

### Feature Importance (Random Forest):
1. 🚬 **Smoking** - Most influential factor (>60% importance)
2. 📅 **Age** - Second most important
3. ⚖️ **BMI** - Third in importance
4. 👶 **Children** - Minor impact
5. ⚧️ **Sex** - Minimal impact
6. 📍 **Region** - Minimal impact

## 🎯 Key Findings

✅ **Smoking is the dominant factor** - Smokers pay substantially higher premiums

✅ **Age and BMI matter** - Older age and higher BMI increase costs, especially for smokers

✅ **Non-linear relationships exist** - Polynomial and tree-based models outperform linear models

✅ **Gender and region have minimal impact** - After controlling for other factors

✅ **Family size shows weak correlation** - Number of children doesn't strongly affect costs

## 🚀 Usage

### Quick Start

```python
# Load the notebook and run all cells
jupyter notebook notebooks/insurance-cost-with-linear-regression.ipynb
```

### Making Predictions

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

# Example: Predict insurance cost for a 35-year-old smoker
age = 35
bmi = 28.5
children = 2
smoker = 1  # 1 for yes, 0 for no

# Create feature array
features = np.array([[age, bmi, children, smoker]])

# Transform with polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
features_poly = poly.fit_transform(features)

# Predict (assuming model is trained)
predicted_cost = model.predict(features_poly)
print(f"Estimated Annual Cost: ${predicted_cost[0]:,.2f}")
```

## 🔮 Future Improvements

- [ ] Add more features (medical history, occupation, etc.)
- [ ] Implement deep learning models for better accuracy
- [ ] Create a web interface for easy predictions
- [ ] Add time-series analysis for cost trends
- [ ] Develop separate models for different demographic groups
- [ ] Implement model deployment using Flask/FastAPI
- [ ] Add cross-validation for more robust evaluation
- [ ] Explore ensemble methods combining multiple models

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**
- GitHub: [@G0dZilLAaA](https://github.com/G0dZilLAaA)
- LinkedIn: [mohit-130-kumawat](https://linkedin.com/in/mohit-130-kumawat)
- Email: kumawatmohit435@gmail.com

## 🙏 Acknowledgments

- Dataset source: [Kaggle - Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- Inspiration from various machine learning projects in healthcare analytics
- Thanks to the open-source community for amazing tools and libraries

## 📧 Contact

For questions or feedback, please open an issue or reach out via email.

---

⭐ **If you found this project helpful, please consider giving it a star!** ⭐

---

**Note**: This project is for educational purposes and should not be used as the sole basis for insurance pricing decisions.# insurance_prediction
