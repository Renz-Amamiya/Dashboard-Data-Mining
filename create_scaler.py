"""Script untuk membuat scaler dari data raw"""
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

def compute_features(df):
    """Compute all 54 features from raw data"""
    features = []
    
    for idx, row in df.iterrows():
        sex = row['Sex']
        age = float(row['Age'])
        birth_weight = float(row['Birth_Weight'])
        birth_length = float(row['Birth_Length'])
        body_weight = float(row['Body_Weight'])
        body_length = float(row['Body_Length'])
        asi = row['ASI_Eksklusif']
        
        # Encoded features
        sex_encoded = 1.0 if sex == "Male" else 0.0
        asi_encoded = 1.0 if asi == "Yes" else 0.0
        
        # Derived features
        bmi = body_weight / ((body_length / 100.0) ** 2) if body_length > 0 else 0.0
        weight_growth = body_weight - birth_weight
        length_growth = body_length - birth_length
        weight_growth_rate = weight_growth / age if age > 0 else 0.0
        length_growth_rate = length_growth / age if age > 0 else 0.0
        weight_per_age = body_weight / age if age > 0 else 0.0
        length_per_age = body_length / age if age > 0 else 0.0
        
        # Binary indicators
        low_birth_weight = 1.0 if birth_weight < 2.5 else 0.0
        very_low_birth_weight = 1.0 if birth_weight < 1.5 else 0.0
        short_birth_length = 1.0 if birth_length < 48.0 else 0.0
        
        # Birth weight category
        if birth_weight < 1.5:
            birth_weight_category = 0
        elif birth_weight < 2.5:
            birth_weight_category = 1
        elif birth_weight < 4.0:
            birth_weight_category = 2
        else:
            birth_weight_category = 3
        
        # Z-scores (simplified)
        age_months = age
        expected_length = 49 + (age_months * 2.5) if age_months <= 24 else 49 + (24 * 2.5) + ((age_months - 24) * 0.5)
        length_z_score = (body_length - expected_length) / 3.0
        
        expected_weight = 3.2 + (age_months * 0.4) if age_months <= 12 else 3.2 + (12 * 0.4) + ((age_months - 12) * 0.2)
        weight_z_score = (body_weight - expected_weight) / 1.5
        
        expected_weight_for_length = (body_length / 100) * 15
        wfl_z_score = (body_weight - expected_weight_for_length) / 1.5
        
        # WHO indicators
        stunting_who = 1.0 if length_z_score < -2 else 0.0
        severe_stunting = 1.0 if length_z_score < -3 else 0.0
        underweight = 1.0 if weight_z_score < -2 else 0.0
        wasting = 1.0 if wfl_z_score < -2 else 0.0
        overweight = 1.0 if wfl_z_score > 2 else 0.0
        
        # Interaction features
        asi_weight_growth = weight_growth * asi_encoded
        asi_length_growth = length_growth * asi_encoded
        asi_weight_growth_rate = weight_growth_rate * asi_encoded
        sex_weight_growth = weight_growth * sex_encoded
        sex_length_growth = length_growth * sex_encoded
        lbw_weight_growth = weight_growth * low_birth_weight
        lbw_length_growth = length_growth * low_birth_weight
        
        # Nutritional stress
        nutritional_stress = max(0, -weight_z_score) * max(0, -length_z_score)
        
        # Velocity features
        weight_velocity = weight_growth / age if age > 0 else 0.0
        length_velocity = length_growth / age if age > 0 else 0.0
        catch_up_growth = 1.0 if (low_birth_weight == 1 and weight_growth > (age * 0.5)) else 0.0
        
        # Log transformations
        log_body_weight = np.log(body_weight + 1) if body_weight > 0 else 0.0
        log_body_length = np.log(body_length + 1) if body_length > 0 else 0.0
        log_birth_weight = np.log(birth_weight + 1) if birth_weight > 0 else 0.0
        log_birth_length = np.log(birth_length + 1) if birth_length > 0 else 0.0
        log_bmi = np.log(bmi + 1) if bmi > 0 else 0.0
        
        # Age category
        if age <= 6:
            age_category = 0
        elif age <= 12:
            age_category = 1
        elif age <= 24:
            age_category = 2
        else:
            age_category = 3
        
        # Additional derived features
        age_years = age / 12.0
        weight_ratio = body_weight / birth_weight if birth_weight > 0 else 0.0
        length_ratio = body_length / birth_length if birth_length > 0 else 0.0
        bmi_to_age = bmi / age if age > 0 else 0.0
        age_squared = age ** 2
        bmi_squared = bmi ** 2
        weight_growth_squared = weight_growth ** 2
        
        # Percentile features
        weight_percentile = min(100, max(0, (weight_z_score + 3) / 6 * 100))
        length_percentile = min(100, max(0, (length_z_score + 3) / 6 * 100))
        bmi_percentile = min(100, max(0, (wfl_z_score + 3) / 6 * 100))
        length_z_percentile = min(100, max(0, (length_z_score + 3) / 6 * 100))
        
        feature_row = [
            sex_encoded, asi_encoded, age, birth_weight, birth_length,
            body_weight, body_length, bmi, weight_growth, length_growth,
            weight_growth_rate, length_growth_rate, weight_per_age, length_per_age,
            low_birth_weight, very_low_birth_weight, short_birth_length, birth_weight_category,
            length_z_score, weight_z_score, wfl_z_score,
            stunting_who, severe_stunting, underweight, wasting, overweight,
            asi_weight_growth, asi_length_growth, asi_weight_growth_rate,
            sex_weight_growth, sex_length_growth, lbw_weight_growth, lbw_length_growth,
            nutritional_stress, weight_velocity, length_velocity, catch_up_growth,
            log_body_weight, log_body_length, log_birth_weight, log_birth_length, log_bmi,
            age_category, age_years, weight_ratio, length_ratio, bmi_to_age,
            age_squared, bmi_squared, weight_growth_squared,
            weight_percentile, length_percentile, bmi_percentile, length_z_percentile,
        ]
        features.append(feature_row)
    
    return np.array(features)

# Load raw data
print("Loading raw data...")
df_raw = pd.read_csv('dataset_stunting_balanced.csv')
print(f"Raw data shape: {df_raw.shape}")

# Compute features
print("Computing features...")
X = compute_features(df_raw)
print(f"Features shape: {X.shape}")

# Fit scaler
print("Fitting StandardScaler...")
scaler = StandardScaler()
scaler.fit(X)

print(f"Scaler mean sample: {scaler.mean_[:5]}")
print(f"Scaler scale sample: {scaler.scale_[:5]}")

# Feature names
feature_names = [
    'Sex_Encoded', 'ASI_Eksklusif_Encoded', 'Age', 'Birth_Weight', 'Birth_Length',
    'Body_Weight', 'Body_Length', 'BMI', 'Weight_Growth', 'Length_Growth',
    'Weight_Growth_Rate', 'Length_Growth_Rate', 'Weight_per_Age', 'Length_per_Age',
    'Low_Birth_Weight', 'Very_Low_Birth_Weight', 'Short_Birth_Length', 'Birth_Weight_Category',
    'Length_for_Age_Z_Score', 'Weight_for_Age_Z_Score', 'Weight_for_Length_Z_Score',
    'Stunting_WHO_Indicator', 'Severe_Stunting', 'Underweight', 'Wasting', 'Overweight',
    'ASI_Weight_Growth', 'ASI_Length_Growth', 'ASI_Weight_Growth_Rate',
    'Sex_Weight_Growth', 'Sex_Length_Growth', 'LBW_Weight_Growth', 'LBW_Length_Growth',
    'Nutritional_Stress', 'Weight_Velocity', 'Length_Velocity', 'Catch_Up_Growth',
    'Log_Body_Weight', 'Log_Body_Length', 'Log_Birth_Weight', 'Log_Birth_Length', 'Log_BMI',
    'Age_Category_WHO', 'Age_Years', 'Weight_Ratio_to_Birth', 'Length_Ratio_to_Birth',
    'BMI_to_Age_Ratio', 'Age_Squared', 'BMI_Squared', 'Weight_Growth_Squared',
    'Weight_Percentile', 'Length_Percentile', 'BMI_Percentile', 'Length_Z_Score_Percentile',
]

# Save scaler
scaler_data = {
    'scaler': scaler,
    'feature_names': feature_names,
    'mean': scaler.mean_,
    'scale': scaler.scale_
}

with open('feature_scaler.pkl', 'wb') as f:
    pickle.dump(scaler_data, f)
    
print("Scaler saved to feature_scaler.pkl")

# Test with a sample
print("\nTesting with sample data...")
test_row = pd.DataFrame([{
    'Sex': 'Male',
    'Age': 41,
    'Birth_Weight': 2,
    'Birth_Length': 45,
    'Body_Weight': 13,
    'Body_Length': 85,
    'ASI_Eksklusif': 'No'
}])
X_test = compute_features(test_row)
X_test_scaled = scaler.transform(X_test)
print(f"Original features: {X_test[0][:5]}")
print(f"Scaled features: {X_test_scaled[0][:5]}")
