import pandas as pd
import joblib
import numpy as np
from datetime import datetime

def predict_storage_location():
    saved_objects = joblib.load("warehouse_optimization_model.pkl")
    model = saved_objects['model']
    scaler = saved_objects['scaler']
    label_encoders = saved_objects['label_encoders']
    assigned_storage_map = saved_objects.get('assigned_storage_map', {})  # Load assigned storage mapping

    package_data = pd.read_csv("package_dataset.csv")
    warehouse_data = pd.read_csv("warehouse_dataset.csv")

    print("Available Package IDs:")
    print(package_data['Package ID'].sample(10).tolist())  # Show sample package IDs

    num_packages = int(input("Enter the number of packages you want to check: "))
    selected_packages = input("Enter the Package IDs (comma-separated): ").split(',')
    selected_packages = [pkg.strip() for pkg in selected_packages[:num_packages]]

    for package_id in selected_packages:
        package_info = package_data[package_data['Package ID'] == package_id]
        if package_info.empty:
            print(f"Package ID {package_id} not found.")
            continue

        row = package_info.iloc[0]
        new_package = pd.DataFrame({
            'Dimensions (cm³)': [row['Dimensions (cm³)']],
            'Weight (kg)': [row['Weight (kg)']],
            'Category': [label_encoders['Category'].transform([row['Category']])[0]],
            'Priority': [label_encoders['Priority'].transform([row['Priority']])[0]]
        })

        numerical_features = ['Dimensions (cm³)', 'Weight (kg)']
        new_package[numerical_features] = scaler.transform(new_package[numerical_features])

        # Use assigned storage from training
        predicted_storage = assigned_storage_map.get(package_id, "No Storage Available")
        confidence_scores = model.predict_proba(new_package).max()

        storage_details = warehouse_data[warehouse_data['Storage ID'] == predicted_storage]

        if not storage_details.empty:
            storage_details = storage_details.to_dict(orient='records')[0]
        else:
            storage_details = {
                "Storage ID": predicted_storage,
                "Dimensions (cm³)": "N/A",
                "Weight Capacity (kg)": "N/A",
                "Current Occupancy (%)": "N/A",
                "Distance": "N/A",
                "Message": "Storage details not found in warehouse dataset"
            }

        print("\n=== Package Storage Recommendation ===")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Package ID: {row['Package ID']}")
        print(f"  Dimensions: {row['Dimensions (cm³)']} cm³")
        print(f"  Weight: {row['Weight (kg)']} kg")
        print(f"  Category: {row['Category']}")
        print(f"  Priority: {row['Priority']}")
        print("\nRecommended Storage:")
        print(f"  Storage Location: {predicted_storage}")
        print(f"  Confidence: {confidence_scores:.2%}")
        print("\nStorage Details:")
        for key, value in storage_details.items():
            print(f"  {key}: {value}")
        print("\n--------------------------------------------")

if __name__ == "__main__":
    predict_storage_location()