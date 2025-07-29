import pandas as pd
import numpy as np
import logging
import joblib
import os

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

from utils import visualize

"""
Main configuration and setup script for House Price Prediction project.
Sets up logging, directories, and constants.

"""

# ----------------------- Constants -------------------------------
OUTPUT_DIR = "Outputs"
MODEL_DIR = "Models"
MODEL_FILE = os.path.join(MODEL_DIR,"model.pkl")
PIPELINE_FILE = os.path.join(MODEL_DIR,"pipeline.pkl")

# ----------------------- Directory Setup -------------------------
os.makedirs(OUTPUT_DIR,exist_ok = True)
os.makedirs(MODEL_DIR,exist_ok = True)

# ----------------------- Logging Setup ---------------------------
logging.basicConfig(
    filename = os.path.join(OUTPUT_DIR,"main.log"),
    level = logging.DEBUG,
    format = "%(asctime)s -- %(levelname)s -- %(message)s",
    encoding = "utf-8"
)

# ---------------------- Data Loader Function ----------------------
def load_data(path):
    
    if not os.path.exists(path):
        logging.error(f"❌ {path} Directory doesn't exist !!\n")
        return None
    
    try:
        # -------- File Extension Detection --------
        root,ext = os.path.splitext(path)
        
        if (ext == ".csv"):
            data  = pd.read_csv(path)
        elif (ext == ".json"):
            data = pd.read_json(path)
        elif (ext in [".xlsx",".xls"]):
            data = pd.read_excel(path)
        else:
            logging.error(f"❌ Unsupported File Extension {ext}\n")
            return None

        # -------- Empty Data Check ----------------
        if data.empty:
            logging.warning(f"⚠️ Loaded file is empty \n")
            return None

        logging.info(f"✅ {path} is successfully loaded with Shape : {data.shape}\n")
        return data
    
    except PermissionError:
        logging.error("🚫 Permission Denied !!\n")
    except OSError as e:
        logging.error(f"💻 Os error as {e}\n")
    except Exception as e:
        logging.error(f"❗ Unknown error: {e}\n")
        
    return None

# ---------- Training and test data set generation ------------------
def generate_train_test_data_set(data):
    
    # Create an income category attribute as strata
    data["income_cat"] = pd.cut(
        data["median_income"],
        bins = [0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels = [1,2,3,4,5]
    )
    
    split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
    for train_index, test_index in split.split(data, data["income_cat"]):
        strat_train_set = data.loc[train_index].drop("income_cat", axis = 1)
        strat_test_set = data.loc[test_index].drop("income_cat", axis = 1)    

    return strat_train_set, strat_test_set

# ---------------------- Building Pipelines -----------------------
def build_pipeline(num_attribs, cat_attribs):
    
    # -------- Numerical Pipeline --------------
    num_pipeline = Pipeline([
        ("imputer",SimpleImputer(strategy = "mean")),
        ("scaler",StandardScaler())
    ])
    
    # -------- Categorical Pipeline ------------
    cat_pipeline = Pipeline([
        ("imputer",SimpleImputer(strategy = "most_frequent")),
        ("oneHot",OneHotEncoder(handle_unknown = "ignore"))
    ])
    
    # -------- Full Pipeline -------------------
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)       
    ])

    return full_pipeline

def prepared_data(data):
    
    housing_train_set, housing_test_set = generate_train_test_data_set(data)
    
    # Separating Training Features and labels
    housing_labels = housing_train_set["median_house_value"].copy()
    housing_features = housing_train_set.drop("median_house_value", axis=1)
    
    num_attribs = housing_features.select_dtypes(include=["int64", "float64"]).columns
    cat_attribs = housing_features.select_dtypes(include=["object", "category"]).columns
    
    # Build and apply the pipeline
    pipeline = build_pipeline(num_attribs, cat_attribs)
    housing_prepared = pipeline.fit_transform(housing_features)
    
    return  housing_prepared, housing_labels

# ---------------------- Main Execution ----------------------------
if __name__ == "__main__":
    pass
        
        
        