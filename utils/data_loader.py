"""Fungsi untuk loading dan preprocessing data"""
import streamlit as st
import pandas as pd
from constants import DATASETS


def count_stunting(stunting_series):
    """Hitung jumlah kasus stunting, handle baik numerik maupun string"""
    if stunting_series.empty:
        return 0
    
    # Cek apakah kolom numerik
    if pd.api.types.is_numeric_dtype(stunting_series):
        # Jika numerik, hitung yang == 1 atau > 0
        return int((stunting_series == 1).sum() if (stunting_series == 1).any() else (stunting_series > 0).sum())
    else:
        # Jika string, hitung yang bernilai positif (case-insensitive)
        stunting_series_lower = stunting_series.astype(str).str.lower().str.strip()
        positive_values = ['yes', 'stunting', '1', 'true', 'y']
        return int(stunting_series_lower.isin(positive_values).sum())


@st.cache_data
def load_data():
    """Load dan gabungkan semua dataset menjadi satu"""
    dfs = []
    for file_path in DATASETS:
        try:
            df = pd.read_csv(file_path)
            df = normalize_column_names(df)
            # Tambahkan kolom untuk tracking sumber dataset
            df['Dataset_Source'] = file_path.replace('.csv', '').replace('dataset_', '')
            dfs.append(df)
        except FileNotFoundError:
            st.warning(f"File {file_path} tidak ditemukan, dilewati.")
        except Exception as e:
            st.warning(f"Error loading {file_path}: {str(e)}")
    
    if not dfs:
        st.error("Tidak ada dataset yang berhasil dimuat!")
        return pd.DataFrame()
    
    # Gabungkan semua dataset
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Hapus duplikat jika ada (berdasarkan kolom utama)
    key_columns = [c for c in ['Sex', 'Age', 'Birth_Weight', 'Birth_Length', 'Body_Weight', 'Body_Length', 'Stunting'] if c in combined_df.columns]
    if key_columns:
        combined_df = combined_df.drop_duplicates(subset=key_columns, keep='first')
    
    return combined_df


def normalize_column_names(df):
    """Normalisasi nama kolom untuk kompatibilitas antar dataset"""
    df = df.copy()
    # Mapping kolom untuk dataset yang sudah diproses
    if 'Sex_Encoded' in df.columns:
        df['Sex'] = df['Sex_Encoded'].map({0: 'Male', 1: 'Female'})
    if 'ASI_Eksklusif_Encoded' in df.columns:
        df['ASI_Eksklusif'] = df['ASI_Eksklusif_Encoded'].map({0: 'No', 1: 'Yes'})
    return df


def create_stunting_label(df):
    """Buat label stunting untuk visualisasi"""
    if 'Stunting' in df.columns:
        df = df.copy()
        # Handle baik numerik maupun string
        if pd.api.types.is_numeric_dtype(df['Stunting']):
            df['Stunting_Label'] = df['Stunting'].map({0: 'Tidak Stunting', 1: 'Stunting'})
        else:
            # Handle string values
            stunting_lower = df['Stunting'].astype(str).str.lower().str.strip()
            positive_values = ['yes', 'stunting', '1', 'true', 'y']
            df['Stunting_Label'] = stunting_lower.apply(
                lambda x: 'Stunting' if x in positive_values else 'Tidak Stunting'
            )
    return df


def create_crosstab_melted(df, index_col, value_col='Stunting'):
    """Helper untuk membuat crosstab yang sudah di-melt"""
    crosstab = pd.crosstab(df[index_col], df[value_col])
    crosstab_reset = crosstab.reset_index()
    
    # Tentukan value_vars berdasarkan tipe data
    if pd.api.types.is_numeric_dtype(df[value_col]):
        value_vars = [0, 1]
        label_map = {0: 'Tidak Stunting', 1: 'Stunting'}
    else:
        # Ambil unique values dari kolom
        unique_values = df[value_col].unique()
        value_vars = list(unique_values)
        # Buat label map untuk string values
        positive_values = ['yes', 'stunting', '1', 'true', 'y']
        label_map = {
            val: 'Stunting' if str(val).lower().strip() in positive_values else 'Tidak Stunting'
            for val in unique_values
        }
    
    melted = pd.melt(
        crosstab_reset,
        id_vars=[index_col],
        value_vars=value_vars,
        var_name=value_col,
        value_name='Jumlah'
    )
    melted['Stunting_Label'] = melted[value_col].map(label_map)
    return melted

