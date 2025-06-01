import pandas as pd
import numpy as np
import time

# Use time as a seed for different random numbers each run
np.random.seed(int(time.time()))  # Seed based on current time

# Define dataset size
num_packages = 100
num_storage_units = 100

# Generate Package Dataset
package_data = pd.DataFrame({
    "Package ID": [f'P{i+1}' for i in range(num_packages)],
    "Dimensions (cm³)": np.random.randint(1500, 50000, size=num_packages),
    "Weight (kg)": np.random.randint(1, 50, size=num_packages),
    "Category": np.random.choice(["Fragile", "Non-Fragile"], size=num_packages),
    "Priority": np.random.choice(["High", "Low"], size=num_packages, p=[0.3, 0.7])  # Excludes "Medium" priority
})

# Generate Warehouse Storage Dataset
warehouse_data = pd.DataFrame({
    "Storage ID": [f'S{i+1}' for i in range(num_storage_units)],
    "Dimensions (cm³)": np.random.randint(30000, 100000, size=num_storage_units),
    "Weight Capacity (kg)": np.random.randint(150, 200, size=num_storage_units),
    "Current Occupancy (%)": np.random.randint(50, 100, size=num_storage_units),
    "Distance": np.random.choice(["Near", "Far"], size=num_storage_units, p=[0.5, 0.5])  # Balanced
})

# Save the datasets
package_data.to_csv("package_dataset.csv", index=False)
warehouse_data.to_csv("warehouse_dataset.csv", index=False)

print("Datasets generated and saved as package_dataset.csv and warehouse_dataset.csv")