# 🏡 California Housing Price Prediction

## 🔍 Overview
This project aims to analyze housing data from California and predict median house values using multiple regression models. It focuses on understanding the data through visualizations and selecting the best-performing model for accurate predictions.

## 📚 Project Pipeline
1. 📥 Data Loading and Preprocessing
2. 📊 Exploratory Data Analysis (EDA)
3. 🧮 Correlation Analysis
4. 🧠 Model Training and Selection
5. 📈 Model Evaluation
6. 🔮 Prediction and Result Export

## 📁 Project Structure
California-House-Price-Prediction/
│
├── Data/
│   └── housing.csv               # Raw dataset (input)
│
├── Outputs/
│   ├── output.csv                # Predictions from the final model
│   ├── main.log                  # Logs from main.py
│   ├── visualize.log             # Logs from data analysis (EDA)
│   ├── model_selection.log       # Logs from model selection process
│   ├── histogram.png             # Histogram of all features
│   ├── scatter_matrix.png        # Scatter matrix for feature relationships
│   ├── geographical_scatter.png  # House price vs location scatter plot
│
├── utils/
|   |── __init__.py               # for helping purposes, contains all methods
│   ├── visualize.py              # EDA, correlation, visualization logic
│   └── model_select.py           # ML model training, testing, and evaluation
│
├── main.py                       # Main driver script (entry point)
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation

## ⚙️ Installation
***Clone the repository:***
git clone https://github.com/your-username/california-house-price-prediction.git
cd california-house-price-prediction

## Create and activate a virtual environment (optional but recommended):
python -m venv myEnv
***Activate:***
source myEnv/bin/activate        # Mac/Linux
myEnv\Scripts\activate           # Windows

## Install dependencies:
pip install -r requirements.txt

## ▶️ Running the Project
python main.py  # run main.py file
***What happens when you run this:***
Loads and analyzes housing.csv
Saves all EDA plots in Outputs/
Selects the best regression model
Exports predictions to output.csv
Logs all steps in Outputs/*.log
After that there is an input.csv in Data
Run main.py again and it will take input.csv data and will give predictions

## 📊 Features Considered
median_income
total_rooms
total_bedrooms
housing_median_age
population
households
ocean_proximity (categorical)
and more...

## 🧠 Models Trained
Linear Regression
Decision Tree Regressor
✅ Random Forest Regressor (Best performance based on RMSE)

## 📈 Final Output
Best model selected and predictions exported
Data insights through correlation heatmaps and visualizations
Metrics evaluated: MSE, RMSE score

## 🛠️ Technologies Used
Language: Python 🐍
Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Logging

## Conclusion: California Housing Price Prediction Project
In this project, we built a complete machine learning pipeline to predict California housing prices using various regression algorithms. 
**We started by:**
***Loading and preprocessing*** the dataset ( housing.csv ) with careful treatment of missing values, scaling, and encoding using a custom pipeline.

***Stratified splitting*** was used to maintain income category distribution
between train and test sets.

**We trained and evaluated multiple algorithms including:**
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

Through cross-validation, we found that Random Forest performed the best,
offering the lowest RMSE and most stable results.

Finally, we built a script that:
Trains the Random Forest model and saves it using joblib .
Uses an if-else logic to skip retraining if the model exists.
Applies the trained model to new data ( input.csv ) to predict
median_house_value , storing results in output.csv .

This pipeline ensures that predictions are accurate, efficient, and ready for
production deployment.


## ✍️ Author
Vinod Kumar Prajapat + chatgpt
📧 ***kaumarvinod08@gmail.com***
🔗 LinkedIn Profile :- ***https://www.linkedin.com/in/vinod-kumar-prajapatpat-v9the/***