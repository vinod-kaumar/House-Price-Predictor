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

from utils import model_select, visualize

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
        logging.error(f"❌ {path} Directory doesn't exist !!")
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
            logging.error(f"❌ Unsupported File Extension {ext}")
            return None

        # -------- Empty Data Check ----------------
        if data.empty:
            logging.warning(f"⚠️ Loaded file is empty ")
            return None

        logging.info(f"✅ {path} is successfully loaded with Shape : {data.shape}")
        return data
    
    except PermissionError:
        logging.error("🚫 Permission Denied !!")
    except OSError as e:
        logging.error(f"💻 Os error as {e}")
    except Exception as e:
        logging.error(f"❗ Unknown error: {e}")
        
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

# ---------------------- Main Execution ----------------------------
if __name__ == "__main__":
    
    # ---------- Training Phase ------------------
    if not os.path.exists(MODEL_FILE):
        
        # -------- Loading Data Set --------------
        folder = "Data"
        filename = "housing.csv"
        path = os.path.join(folder,filename)
        data = load_data(path)
        
        if data is not None:
            
            logging.info("------------------------------------------------")
            logging.info("Starting EDA and Data visualization...\n")
            logging.info("I'm inside utils/visualize.py...")
            
            visualize.basic_info(data)
            visualize.corr_matrix(data)
            visualize.geographical_scatter_plot(data)
            visualize.scatter_matrix_plot(data)
            visualize.plot_histogram(data)
            
            logging.info("EDA and Data Visualization Done ✅\n")
            
            logging.info("------------------------------------------------")
            logging.info("✅ Starting train-test data split.")
            housing_train_set, housing_test_set = generate_train_test_data_set(data)
            
            # Saving test set for testing the model
            os.makedirs("Data",exist_ok = True)
            path = os.path.join("Data","input.csv")
            housing_test_set.to_csv(path, index = False)
            logging.info("Test Set Saved in Data/input.csv for testing the model")
            
            # Separating Training Features and labels
            housing_labels = housing_train_set["median_house_value"].copy()
            housing_features = housing_train_set.drop("median_house_value", axis=1)
            
            logging.info("✅ Training features and labels separated successfully.")
            
            # Separate numerical and categorical attributes
            num_attribs = housing_features.select_dtypes(include=["int64", "float64"]).columns
            cat_attribs = housing_features.select_dtypes(include=["object", "category"]).columns

            # Log the attribute groups (debugging info)
            logging.debug(f"Numerical attributes: {list(num_attribs)}")
            logging.debug(f"Categorical attributes: {list(cat_attribs)}")

            # Build and apply the pipeline
            pipeline = build_pipeline(num_attribs, cat_attribs)
            housing_prepared = pipeline.fit_transform(housing_features)

            # ✅ Success log
            logging.info("✅ Housing data is successfully prepared for model training.")
            
            logging.info(f"Housing prepared shape: {housing_prepared.shape}\n")
            
            logging.info("------------------------------------------------")
            logging.info("Selecting best Model To be trained --> model_select.py")
            model = model_select.best_model(housing_prepared, housing_labels)
            logging.info("best model is returned and ready for being trained\n")
            
            # Training the model
            logging.info("------------------------------------------------")
            logging.info("Training The model")
            
            # model = RandomForestRegressor(random_state = 42)
            model.fit(housing_prepared, housing_labels)
            
            # Save model and pipeline
            logging.info("Saving model and pipeline")
            joblib.dump(model, MODEL_FILE)
            joblib.dump(pipeline, PIPELINE_FILE)
            
            logging.info("✅ Model trained and saved.")
            
        else:
            logging.warning("⚠️ Data is None. Skipping train-test split.")

    
    # ---------- Inference Phase ------------------
    else:
        # ---------------- Load Saved Model and Pipeline ----------------
        try:
            model = joblib.load(MODEL_FILE)        
            pipeline = joblib.load(PIPELINE_FILE)  
            logging.info("----------------------------------------")
            logging.info("✅ Model and pipeline loaded successfully")
        except Exception as e:
            logging.error(f"❌ Failed to load model or pipeline: {e}")
            raise

        # ---------------- Load Input Data for Prediction ----------------
        input_file_path = "Data/input.csv"

        try:
            input_data = pd.read_csv(input_file_path)
            logging.info("✅ Input data loaded for prediction")
        except Exception as e:
            logging.error(f"❌ Failed to load input data: {e}")
            raise

        # ---------------- Apply Pipeline and Predict ----------------
        try:
            transformed_input = pipeline.transform(input_data)
            predictions = model.predict(transformed_input)
            input_data["median_house_value"] = predictions
            output_path = "Outputs/output.csv"
            input_data.to_csv(output_path, index=False)
            logging.info(f"✅ Predictions generated and saved to {output_path}")
            print(input_data[["median_house_value"]].head())
        except Exception as e:
            logging.error(f"❌ Prediction failed: {e}")
            raise
        
        
        