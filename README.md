# Fish Weight Prediction using Machine Learning 🐟📊

## 📌 Project Overview
This project aims to predict the weight of fish based on multiple features using different machine learning models. The dataset used comes from a **Fish Market** dataset, containing species information and various dimensional measurements.

## 🚀 Models Implemented
1. **Linear Regression**
2. **Decision Tree Regressor**
3. **Random Forest Regressor** (Best-performing model 🎯)
4. **Support Vector Regressor (SVR)**

## 🔍 Challenges Faced
- **Feature Engineering**: Encoding categorical data (Species) and handling multicollinearity.
- **Model Selection**: Trying different models and comparing performance.
- **Hyperparameter Tuning**: Using GridSearchCV to optimize Random Forest.
- **Scaling Issues**: Needed MinMax scaling for SVR but resulted in poor performance.

## 📊 Model Performance
| Model | R² Score |
|--------|---------|
| Linear Regression | ~0.95 |
| Decision Tree | ~0.97 |
| **Random Forest (Tuned)** | **~0.94** |
| SVR | **Poor Performance (-0.10)** |

## 🛠 Tech Stack
- **Python** 🐍
- **Pandas, NumPy** 📊
- **Scikit-Learn** 🤖
- **Seaborn, Matplotlib** 📈

## 📂 Dataset
The dataset contains the following features:
- `Species` (Categorical, encoded)
- `Length1`, `Length2`, `Length3` (Fish length measurements)
- `Height`, `Width` (Other fish dimensions)
- `Weight` (Target variable)

## 🔧 Setup & Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/fish-weight-prediction.git
cd fish-weight-prediction

# Install dependencies
pip install -r requirements.txt

# Run the model script
python model.py
```

## 📌 Key Learnings
- **Random Forest** gave the best results after hyperparameter tuning.
- **SVR** performed poorly due to improper assumptions about data distribution.
- **Feature importance analysis** helped in understanding which parameters influenced weight prediction the most.

## 🤝 Connect With Me
If you find this project interesting or want to collaborate, feel free to connect!
- **X (Twitter):** [@Sourav Kumar](https://x.com/souravkumarr73)
- **LinkedIn:** [Sourav Kumar](https://www.linkedin.com/in/sourav-kumar-30141b174/)

⭐ Don't forget to **star** this repo if you found it helpful! 🚀
