from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

import pandas as pd
import numpy as np
import logging
import os

# Ensure Outputs directory exists
os.makedirs("Outputs", exist_ok=True)

# Setup logging
logging.basicConfig(
    filename="Outputs/main.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8"
)

def best_model(housing_features, housing_labels):
    logging.info("🔍 Selecting best model using cross-validation...")

    # Models
    lin_reg = LinearRegression()
    tree_reg = DecisionTreeRegressor(random_state=42)
    forest_reg = RandomForestRegressor(random_state=42)

    """
    Training RMSE only shows how well the model fits the training data. It does not
    tell us how well it will perform on unseen data. In fact, the Decision Tree and
    Random Forest may overfit, leading to very low training error but poor
    generalization.

    Cross-Validation: A Better Evaluation Strategy
    """


    # Cross-Validation RMSE
    lin_rmses = -cross_val_score(lin_reg, housing_features, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
    
    tree_rmses = -cross_val_score(tree_reg, housing_features, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
    
    forest_rmses = -cross_val_score(forest_reg, housing_features, housing_labels, scoring="neg_root_mean_squared_error", cv=10)

    # Logging
    logging.info(f"Linear Regression CV RMSEs: {lin_rmses}")
    logging.info("Performance Summary (Linear Regression):")
    logging.info(pd.Series(lin_rmses).describe())

    logging.info(f"Decision Tree CV RMSEs: {tree_rmses}")
    logging.info("Performance Summary (Decision Tree):")
    logging.info(pd.Series(tree_rmses).describe())

    logging.info(f"Random Forest CV RMSEs: {forest_rmses}")
    logging.info("Performance Summary (Random Forest):")
    logging.info(pd.Series(forest_rmses).describe())

    # Compare
    avg_rmse = {
        "Linear Regression": lin_rmses.mean(),
        "Decision Tree": tree_rmses.mean(),
        "Random Forest": forest_rmses.mean()
    }

    best_model_name = min(avg_rmse, key=avg_rmse.get)
    logging.info(f"✅ Best model based on average CV RMSE: {best_model_name}")

    if best_model_name == "Linear Regression":
        lin_reg.fit(housing_features, housing_labels)
        return lin_reg
    elif best_model_name == "Decision Tree":
        tree_reg.fit(housing_features, housing_labels)
        return tree_reg
    else:
        forest_reg.fit(housing_features, housing_labels)
        return forest_reg

if __name__ == "__main__":
    pass
