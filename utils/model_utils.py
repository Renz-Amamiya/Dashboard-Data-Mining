"""Fungsi untuk model dan prediksi"""
import streamlit as st
import os
import numpy as np
from constants import MODEL_PATH

# Import untuk model
try:
    from tensorflow import keras
    import tensorflow as tf
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    keras = None
    tf = None


def _check_h5_structure(model_path):
    """Cek struktur file HDF5 untuk menentukan apakah hanya berisi weights"""
    try:
        import h5py
        with h5py.File(model_path, 'r') as f:
            keys = list(f.keys())
            # Model lengkap biasanya punya 'model_config' atau 'config'
            # Weights saja biasanya hanya punya layer names atau 'model_weights'
            has_config = 'model_config' in keys or 'config' in keys
            has_weights = 'model_weights' in keys or any('weight' in str(k).lower() for k in keys)
            return has_config, has_weights, keys
    except Exception:
        return None, None, []


@st.cache_resource
def load_model(model_path):
    """Load model H5 dengan berbagai metode fallback, termasuk format pickle dalam HDF5"""
    import pickle
    import h5py
    
    if not os.path.exists(model_path):
        st.error(f"⚠️ File model {model_path} tidak ditemukan.")
        return None
    
    errors = []
    
    # Cek struktur file HDF5
    has_config, has_weights, h5_keys = _check_h5_structure(model_path)
    
    # Metode 1: Load dari format pickle dalam HDF5 (format yang digunakan notebook)
    try:
        with h5py.File(model_path, 'r') as hf:
            if 'model' in hf.keys():
                model_data = hf['model'][()]
                # Cek apakah ini model yang di-pickle
                if isinstance(model_data, bytes):
                    model = pickle.loads(model_data)
                    st.success(f"✅ Model berhasil dimuat dari format HDF5+pickle")
                    return model
                else:
                    # Coba convert ke bytes dan unpickle
                    model = pickle.loads(model_data.tobytes())
                    st.success(f"✅ Model berhasil dimuat dari format HDF5+pickle (tobytes)")
                    return model
    except Exception as e1:
        errors.append(f"Metode 1 (HDF5+pickle): {str(e1)}")
    
    # Hanya lanjut dengan metode Keras jika TensorFlow tersedia
    if MODEL_AVAILABLE:
        # Metode 2: Load model lengkap Keras (default)
        try:
            model = keras.models.load_model(model_path, compile=False)
            st.success(f"✅ Model berhasil dimuat menggunakan metode Keras default")
            return model
        except Exception as e2:
            errors.append(f"Metode 2 (Keras default): {str(e2)}")
        
        # Metode 3: Load dengan safe_mode=False (untuk TensorFlow 2.16+)
        try:
            model = keras.models.load_model(model_path, compile=False, safe_mode=False)
            st.success(f"✅ Model berhasil dimuat menggunakan safe_mode=False")
            return model
        except Exception as e3:
            errors.append(f"Metode 3 (safe_mode=False): {str(e3)}")
        
        # Metode 4: Coba dengan tf.keras langsung
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            st.success(f"✅ Model berhasil dimuat menggunakan tf.keras")
            return model
        except Exception as e4:
            errors.append(f"Metode 4 (tf.keras): {str(e4)}")
        
        # Metode 5: Coba dengan tf.keras dan safe_mode=False
        try:
            model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
            st.success(f"✅ Model berhasil dimuat menggunakan tf.keras dengan safe_mode=False")
            return model
        except Exception as e5:
            errors.append(f"Metode 5 (tf.keras safe_mode=False): {str(e5)}")
    else:
        errors.append("TensorFlow tidak tersedia - metode Keras dilewati")
    
    # Jika semua metode gagal, tampilkan error yang lebih informatif
    tf_version = tf.__version__ if (MODEL_AVAILABLE and tf) else 'Tidak terinstall'
    file_size_mb = os.path.getsize(model_path) / (1024*1024)
    
    # Tampilkan error dengan format yang lebih rapi
    st.error(f"**❌ Error loading model: {model_path}**")
    
    with st.expander("📋 Detail Error", expanded=True):
        for i, err in enumerate(errors, 1):
            st.text(f"{i}. {err}")
    
    with st.expander("ℹ️ Informasi File & Sistem", expanded=False):
        st.write(f"**Path:** `{os.path.abspath(model_path)}`")
        st.write(f"**Ukuran:** {file_size_mb:.2f} MB")
        st.write(f"**Keys HDF5:** {', '.join(h5_keys[:10]) if h5_keys else 'Tidak dapat dibaca'}")
        st.write(f"**Memiliki config:** {has_config if has_config is not None else 'Tidak diketahui'}")
        st.write(f"**Memiliki weights:** {has_weights if has_weights is not None else 'Tidak diketahui'}")
        st.write(f"**Versi TensorFlow:** {tf_version}")
    
    st.info("""
    **💡 Solusi yang bisa dicoba:**
    
    1. **Pastikan file model valid dan tidak corrupt**
    
    2. **Install TensorFlow jika belum:**
       ```bash
       pip install tensorflow
       ```
    
    3. **Jika masih error, model mungkin dibuat dengan versi TensorFlow yang berbeda**
    """)
    
    return None


def preprocess_input(sex, age, birth_weight, birth_length, body_weight, body_length, asi):
    """
    Preprocess input untuk prediksi model sklearn.
    Menghitung semua 54 fitur yang diperlukan berdasarkan input pengguna.
    
    Fitur yang dihitung:
    0. Sex_Encoded, 1. ASI_Eksklusif_Encoded, 2. Age, 3. Birth_Weight, 4. Birth_Length,
    5. Body_Weight, 6. Body_Length, 7. BMI, 8. Weight_Growth, 9. Length_Growth,
    10. Weight_Growth_Rate, 11. Length_Growth_Rate, 12. Weight_per_Age, 13. Length_per_Age,
    14. Low_Birth_Weight, 15. Very_Low_Birth_Weight, 16. Short_Birth_Length, 17. Birth_Weight_Category,
    18. Length_for_Age_Z_Score, 19. Weight_for_Age_Z_Score, 20. Weight_for_Length_Z_Score,
    21. Stunting_WHO_Indicator, 22. Severe_Stunting, 23. Underweight, 24. Wasting, 25. Overweight,
    26. ASI_Weight_Growth, 27. ASI_Length_Growth, 28. ASI_Weight_Growth_Rate, 29. Sex_Weight_Growth,
    30. Sex_Length_Growth, 31. LBW_Weight_Growth, 32. LBW_Length_Growth, 33. Nutritional_Stress,
    34. Weight_Velocity, 35. Length_Velocity, 36. Catch_Up_Growth, 37. Log_Body_Weight,
    38. Log_Body_Length, 39. Log_Birth_Weight, 40. Log_Birth_Length, 41. Log_BMI,
    42. Age_Category_WHO, 43. Age_Years, 44. Weight_Ratio_to_Birth, 45. Length_Ratio_to_Birth,
    46. BMI_to_Age_Ratio, 47. Age_Squared, 48. BMI_Squared, 49. Weight_Growth_Squared,
    50. Weight_Percentile, 51. Length_Percentile, 52. BMI_Percentile, 53. Length_Z_Score_Percentile
    """
    # Convert inputs to float
    age = float(age)
    birth_weight = float(birth_weight)
    birth_length = float(birth_length)
    body_weight = float(body_weight)
    body_length = float(body_length)
    
    # Encoded features
    sex_encoded = 1.0 if sex == "Male" else 0.0
    asi_encoded = 1.0 if asi == "Yes" else 0.0
    
    # Basic features (langsung dari input)
    # Age, Birth_Weight, Birth_Length, Body_Weight, Body_Length sudah ada
    
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
    
    # Birth weight category (0=very low, 1=low, 2=normal, 3=high)
    if birth_weight < 1.5:
        birth_weight_category = 0
    elif birth_weight < 2.5:
        birth_weight_category = 1
    elif birth_weight < 4.0:
        birth_weight_category = 2
    else:
        birth_weight_category = 3
    
    # Z-scores (menggunakan WHO standards approximation)
    # Ini adalah estimasi sederhana, nilai sebenarnya memerlukan tabel WHO
    age_months = age
    
    # Simplified Z-score calculations
    # Length-for-age Z-score (LAZ)
    expected_length = 49 + (age_months * 2.5) if age_months <= 24 else 49 + (24 * 2.5) + ((age_months - 24) * 0.5)
    length_z_score = (body_length - expected_length) / 3.0
    
    # Weight-for-age Z-score (WAZ)
    expected_weight = 3.2 + (age_months * 0.4) if age_months <= 12 else 3.2 + (12 * 0.4) + ((age_months - 12) * 0.2)
    weight_z_score = (body_weight - expected_weight) / 1.5
    
    # Weight-for-length Z-score (WLZ)
    expected_weight_for_length = (body_length / 100) * 15  # Simplified
    wfl_z_score = (body_weight - expected_weight_for_length) / 1.5
    
    # WHO indicators based on Z-scores
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
    
    # Nutritional stress indicator
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
    
    # Age category WHO (0=0-6, 1=6-12, 2=12-24, 3=24-60)
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
    
    # Percentile features (simplified, using rank-based approximation)
    # Dalam praktiknya, ini dihitung dari distribusi data training
    # Untuk prediksi real-time, kita gunakan estimasi berdasarkan Z-score
    weight_percentile = min(100, max(0, (weight_z_score + 3) / 6 * 100))
    length_percentile = min(100, max(0, (length_z_score + 3) / 6 * 100))
    bmi_percentile = min(100, max(0, (wfl_z_score + 3) / 6 * 100))
    length_z_percentile = min(100, max(0, (length_z_score + 3) / 6 * 100))
    
    # Compile all 54 features in the correct order
    features = [
        sex_encoded,           # 0. Sex_Encoded
        asi_encoded,           # 1. ASI_Eksklusif_Encoded
        age,                   # 2. Age
        birth_weight,          # 3. Birth_Weight
        birth_length,          # 4. Birth_Length
        body_weight,           # 5. Body_Weight
        body_length,           # 6. Body_Length
        bmi,                   # 7. BMI
        weight_growth,         # 8. Weight_Growth
        length_growth,         # 9. Length_Growth
        weight_growth_rate,    # 10. Weight_Growth_Rate
        length_growth_rate,    # 11. Length_Growth_Rate
        weight_per_age,        # 12. Weight_per_Age
        length_per_age,        # 13. Length_per_Age
        low_birth_weight,      # 14. Low_Birth_Weight
        very_low_birth_weight, # 15. Very_Low_Birth_Weight
        short_birth_length,    # 16. Short_Birth_Length
        birth_weight_category, # 17. Birth_Weight_Category
        length_z_score,        # 18. Length_for_Age_Z_Score
        weight_z_score,        # 19. Weight_for_Age_Z_Score
        wfl_z_score,           # 20. Weight_for_Length_Z_Score
        stunting_who,          # 21. Stunting_WHO_Indicator
        severe_stunting,       # 22. Severe_Stunting
        underweight,           # 23. Underweight
        wasting,               # 24. Wasting
        overweight,            # 25. Overweight
        asi_weight_growth,     # 26. ASI_Weight_Growth
        asi_length_growth,     # 27. ASI_Length_Growth
        asi_weight_growth_rate,# 28. ASI_Weight_Growth_Rate
        sex_weight_growth,     # 29. Sex_Weight_Growth
        sex_length_growth,     # 30. Sex_Length_Growth
        lbw_weight_growth,     # 31. LBW_Weight_Growth
        lbw_length_growth,     # 32. LBW_Length_Growth
        nutritional_stress,    # 33. Nutritional_Stress
        weight_velocity,       # 34. Weight_Velocity
        length_velocity,       # 35. Length_Velocity
        catch_up_growth,       # 36. Catch_Up_Growth
        log_body_weight,       # 37. Log_Body_Weight
        log_body_length,       # 38. Log_Body_Length
        log_birth_weight,      # 39. Log_Birth_Weight
        log_birth_length,      # 40. Log_Birth_Length
        log_bmi,               # 41. Log_BMI
        age_category,          # 42. Age_Category_WHO
        age_years,             # 43. Age_Years
        weight_ratio,          # 44. Weight_Ratio_to_Birth
        length_ratio,          # 45. Length_Ratio_to_Birth
        bmi_to_age,            # 46. BMI_to_Age_Ratio
        age_squared,           # 47. Age_Squared
        bmi_squared,           # 48. BMI_Squared
        weight_growth_squared, # 49. Weight_Growth_Squared
        weight_percentile,     # 50. Weight_Percentile
        length_percentile,     # 51. Length_Percentile
        bmi_percentile,        # 52. BMI_Percentile
        length_z_percentile,   # 53. Length_Z_Score_Percentile
    ]
    
    features_array = np.array([features], dtype=np.float32)
    
    # Load and apply scaler if available
    scaler_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'feature_scaler.pkl')
    if os.path.exists(scaler_path):
        try:
            import pickle
            with open(scaler_path, 'rb') as f:
                scaler_data = pickle.load(f)
            scaler = scaler_data['scaler']
            features_scaled = scaler.transform(features_array)
            return features_scaled.astype(np.float32)
        except Exception as e:
            # If scaler fails, return unscaled features with warning
            print(f"Warning: Could not apply scaler: {e}")
            return features_array
    else:
        # No scaler found, return unscaled features
        return features_array


def interpret_prediction(prediction):
    """Interpretasi hasil prediksi model"""
    if len(prediction[0]) == 1:
        prob_stunting = max(0.0, min(1.0, float(prediction[0][0])))
        prob_no_stunting = 1.0 - prob_stunting
    else:
        prob_no_stunting = float(prediction[0][0])
        prob_stunting = float(prediction[0][1]) if len(prediction[0]) > 1 else 0.0
        total = prob_no_stunting + prob_stunting
        if total > 0:
            prob_no_stunting /= total
            prob_stunting /= total
    
    threshold = 0.5
    result = "Stunting" if prob_stunting > threshold else "Tidak Stunting"
    return prob_no_stunting, prob_stunting, result

