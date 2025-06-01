import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import warnings

warnings.filterwarnings("ignore")  # Suppress warnings for clean output

def preprocess_data():
    try:
        package_data = pd.read_csv("package_dataset.csv")
        warehouse_data = pd.read_csv("warehouse_dataset.csv")
    except FileNotFoundError:
        print("Error: Dataset files not found. Run generate_dataset.py first.")
        return None, None, None

    label_encoders = {
        'Category': LabelEncoder().fit(package_data['Category']),
        'Priority': LabelEncoder().fit(package_data['Priority']),
        'Distance': LabelEncoder().fit(warehouse_data['Distance'])
    }

    package_data['Category'] = label_encoders['Category'].transform(package_data['Category'])
    package_data['Priority'] = label_encoders['Priority'].transform(package_data['Priority'])
    warehouse_data['Distance'] = label_encoders['Distance'].transform(warehouse_data['Distance'])

    return package_data, warehouse_data, label_encoders

used_storage_ids = set()  # Track used storage IDs
assigned_storage_map = {}  # Track package to storage mapping

def assign_optimal_storage(row, warehouse_data):
    global used_storage_ids, assigned_storage_map

    # Filter storage based on priority
    if row['Priority'] == 0:  # High priority
        suitable_storage = warehouse_data[
            (warehouse_data['Dimensions (cm³)'] >= row['Dimensions (cm³)']) &
            (warehouse_data['Weight Capacity (kg)'] >= row['Weight (kg)']) &
            (warehouse_data['Current Occupancy (%)'] < 90) &
            (warehouse_data['Distance'] == 1) &  # Near storage
            (~warehouse_data['Storage ID'].isin(used_storage_ids))
        ]

        # If no near storage available, consider far storage
        if suitable_storage.empty:
            suitable_storage = warehouse_data[
                (warehouse_data['Dimensions (cm³)'] >= row['Dimensions (cm³)']) &
                (warehouse_data['Weight Capacity (kg)'] >= row['Weight (kg)']) &
                (warehouse_data['Current Occupancy (%)'] < 90) &
                (warehouse_data['Distance'] == 0) &  # Far storage
                (~warehouse_data['Storage ID'].isin(used_storage_ids))
            ]
    else:  # Low priority
        suitable_storage = warehouse_data[
            (warehouse_data['Dimensions (cm³)'] >= row['Dimensions (cm³)']) &
            (warehouse_data['Weight Capacity (kg)'] >= row['Weight (kg)']) &
            (warehouse_data['Current Occupancy (%)'] < 90) &
            (~warehouse_data['Storage ID'].isin(used_storage_ids))
        ]

    if suitable_storage.empty:
        assigned_storage_map[row['Package ID']] = "No Storage Available"
        return "No Storage Available"

    # Introduce randomness to prevent deterministic assignments
    if np.random.rand() < 0.1:  # 10% chance to assign random suitable storage
        selected_storage = suitable_storage.sample(1).iloc[0]['Storage ID']
    else:
        selected_storage = suitable_storage.nsmallest(1, 'Current Occupancy (%)').iloc[0]['Storage ID']

    used_storage_ids.add(selected_storage)  # Mark storage as used
    assigned_storage_map[row['Package ID']] = selected_storage  # Track assignment
    return selected_storage

def train_model():
    package_data, warehouse_data, label_encoders = preprocess_data()
    if package_data is None:
        return

    package_data['Assigned_Storage'] = package_data.apply(lambda row: assign_optimal_storage(row, warehouse_data), axis=1)

    # Remove classes with only one instance to avoid ValueError in stratified split
    class_counts = package_data['Assigned_Storage'].value_counts()
    valid_classes = class_counts[class_counts > 1].index
    filtered_package_data = package_data[package_data['Assigned_Storage'].isin(valid_classes)]

    features = ['Dimensions (cm³)', 'Weight (kg)', 'Category', 'Priority']
    X = filtered_package_data[features].copy()
    y = filtered_package_data['Assigned_Storage']

    scaler = StandardScaler()
    X.loc[:, ['Dimensions (cm³)', 'Weight (kg)']] = scaler.fit_transform(X[['Dimensions (cm³)', 'Weight (kg)']])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    model = RandomForestClassifier(
        n_estimators=10,  # Reduced number of trees
        max_depth=4,  # Shallower trees to reduce model complexity
        min_samples_split=50,  # Require more samples to split
        min_samples_leaf=10,  # Require more samples in leaf nodes
        max_features='sqrt',  # Use fewer features for splits
        random_state=42
    )

    # Apply K-Fold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Model training and testing completed successfully.")
    print(f"Final Model Accuracy: {accuracy:.2f}")
    print(f"Cross-Validation Accuracy Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {np.mean(cv_scores):.2f}")

    joblib.dump({'model': model, 'scaler': scaler, 'label_encoders': label_encoders, 'used_storage_ids': used_storage_ids, 'assigned_storage_map': assigned_storage_map}, 'warehouse_optimization_model.pkl')
    print("Model and preprocessing objects saved successfully.")

    # Create Allocated Storage CSV
    allocated_storage = []
    for idx, row in package_data.iterrows():
        storage_info = warehouse_data[warehouse_data['Storage ID'] == row['Assigned_Storage']]
        if not storage_info.empty:
            storage_info = storage_info.iloc[0]
            remaining_dimension = storage_info['Dimensions (cm³)'] - row['Dimensions (cm³)']
            remaining_weight_capacity = storage_info['Weight Capacity (kg)'] - row['Weight (kg)']
            distance = 'Near' if storage_info['Distance'] == 1 else 'Far'
        else:
            remaining_dimension = "N/A"
            remaining_weight_capacity = "N/A"
            distance = "N/A"

        allocated_storage.append({
            "Package ID": row['Package ID'],
            "Package Dimensions (cm³)": row['Dimensions (cm³)'],
            "Package Weight (kg)": row['Weight (kg)'],
            "Package Category": 'Fragile' if row['Category'] == 0 else 'Non-Fragile',
            "Package Priority": 'High' if row['Priority'] == 0 else 'Low',
            "Storage ID": row['Assigned_Storage'],
            "Remaining Storage Dimensions (cm³)": remaining_dimension,
            "Remaining Weight Capacity (kg)": remaining_weight_capacity,
            "Distance": distance
        })

    allocated_storage_df = pd.DataFrame(allocated_storage)
    allocated_storage_df.to_csv("Allocated_Storage.csv", index=False)
    print("Allocated storage details saved as Allocated_Storage.csv.")

if __name__ == "__main__":
    train_model()