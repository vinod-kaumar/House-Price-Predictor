from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from utils import __init__ as i
import seaborn as sns
import pandas as pd
import logging
import os

# Ensure Outputs directory exists
os.makedirs("Outputs", exist_ok=True)

# Setup logging
logging.basicConfig(
    filename="Outputs/main.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding = "utf-8"
)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

def basic_info(data):
    logging.info("Showing basic info of dataset:")
    logging.info("\n" + str(data.head()))
    logging.info(f"\n{data.info()}")
    logging.info("\n" + str(data.describe()))

def geographical_scatter_plot(data):

    plt.figure(figsize = (12,7))
    
    scatter = plt.scatter(
        x = data["longitude"],
        y = data["latitude"],
        c=data["median_house_value"],
        cmap="plasma",   
        alpha=0.7,
        edgecolor="k",
        linewidth=0.2
    )
    lbar = plt.colorbar(scatter)
    lbar.set_label(label = "Median House Value($)", fontsize = 12)
    plt.xlabel("Longitudes", fontsize=12)
    plt.ylabel("Latitudes", fontsize=12)
    plt.title(
        "California Housing Prices by Location", 
        fontsize=14,
        fontweight='bold'
    )
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig("Outputs/geographical_scatter.png", dpi=300)
    logging.info("📊 Geographical Scatter plot saved to Outputs/geographical_scatter.png")

def corr_matrix(data):
    numeric_data = data.select_dtypes(include=["number"])
    
    cor_mat = numeric_data.corr()
    logging.info("📈 Correlation :\n%s", cor_mat)

def scatter_matrix_plot(data):

    # Attributes to plot
    attributes = ["median_house_value", "median_income", 
                  "total_rooms", "housing_median_age"]

    # Plot scatter matrix
    scatter_matrix(
        data[attributes],
        figsize=(12, 7),
        diagonal='hist',
        alpha=0.6,
        color='#5A9',
        hist_kwds={'bins': 30, 'color': '#FDB813', 'edgecolor': 'black'}
    )

    plt.suptitle("Scatter Matrix of Housing Data", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("Outputs/scatter_matrix.png", dpi=300)
    logging.info("📊 Scatter Matrix plot saved to Outputs/scatter_matrix.png")
    
def plot_histogram(data):
    data.hist(bins=50, figsize=(10, 6))
    plt.tight_layout()
    plt.savefig("Outputs/histogram.png")
    logging.info("📊 Histogram saved to Outputs/histogram.png")

if __name__ == "__main__":
    folder = "Data"
    filename = "housing.csv"
    path = os.path.join(folder, filename)
    
    data = i.load_data(path)
    
    if data is not None:
        # basic_info(data)
        # corr_matrix(data)
        # geographical_scatter_plot(data)
        # scatter_matrix_plot(data)
        # plot_histogram(data)
        pass
    else:
        logging.error("❌ Failed to load data.")
 