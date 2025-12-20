"""Fungsi untuk filter sidebar"""
import streamlit as st
import pandas as pd
from utils.data_loader import count_stunting


def setup_sidebar_filters(df):
    """Setup filter di sidebar dan return filtered dataframe"""
    st.sidebar.title("Navigasi Dashboard")
    st.sidebar.markdown("---")
    
    # Menu navigasi
    page = st.sidebar.radio(
        "Pilih Halaman",
        ["Overview", "Analisis Visual", "Analisis Detail", "Data Explorer", "Prediksi"]
    )
    
    # Filter di sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filter Data")
    
    # Filter berdasarkan jenis kelamin
    if 'Sex' in df.columns:
        sex_options = [
            val for val in df['Sex'].unique() 
            if pd.notna(val) and str(val).strip() != '' 
            and str(val).strip() not in ['0', '1']
        ]
        if sex_options:
            sex_filter = st.sidebar.multiselect(
                "Jenis Kelamin",
                options=sex_options,
                default=sex_options
            )
        else:
            sex_filter = []
    else:
        sex_filter = []
    
    # Filter berdasarkan ASI Eksklusif
    if 'ASI_Eksklusif' in df.columns:
        asi_options = [
            val for val in df['ASI_Eksklusif'].unique() 
            if pd.notna(val) and str(val).strip() != '' 
            and str(val).strip() not in ['0', '1']
        ]
        if asi_options:
            asi_filter = st.sidebar.multiselect(
                "ASI Eksklusif",
                options=asi_options,
                default=asi_options
            )
        else:
            asi_filter = []
    else:
        asi_filter = []
    
    # Filter berdasarkan Stunting (hanya tampilkan jika bukan numeric 0/1)
    if 'Stunting' in df.columns:
        # Cek apakah kolom Stunting adalah numeric dengan nilai 0/1
        if pd.api.types.is_numeric_dtype(df['Stunting']):
            # Jika numeric, jangan tampilkan filter (gunakan semua data)
            stunting_filter = None  # None berarti tidak ada filter
        else:
            # Jika string/categorical, tampilkan filter seperti biasa (tapi filter 0/1)
            stunting_options = [
                val for val in df['Stunting'].unique() 
                if pd.notna(val) and str(val).strip() != '' 
                and str(val).strip() not in ['0', '1']
            ]
            if stunting_options:
                stunting_filter = st.sidebar.multiselect(
                    "Status Stunting",
                    options=stunting_options,
                    default=stunting_options
                )
            else:
                stunting_filter = []
    else:
        stunting_filter = None
    
    # Filter umur
    if 'Age' in df.columns:
        age_range = st.sidebar.slider(
            "Rentang Umur (bulan)",
            min_value=int(df['Age'].min()),
            max_value=int(df['Age'].max()),
            value=(int(df['Age'].min()), int(df['Age'].max()))
        )
    else:
        age_range = (0, 100)
    
    # Terapkan filter
    filtered_df = df.copy()
    if 'Sex' in df.columns:
        filtered_df = filtered_df[filtered_df['Sex'].isin(sex_filter)]
    if 'ASI_Eksklusif' in df.columns:
        filtered_df = filtered_df[filtered_df['ASI_Eksklusif'].isin(asi_filter)]
    if 'Stunting' in df.columns and stunting_filter is not None:
        # Hanya terapkan filter jika stunting_filter bukan None (bukan numeric)
        filtered_df = filtered_df[filtered_df['Stunting'].isin(stunting_filter)]
    if 'Age' in df.columns:
        filtered_df = filtered_df[(filtered_df['Age'] >= age_range[0]) & (filtered_df['Age'] <= age_range[1])]
    
    # Informasi di sidebar
    st.sidebar.markdown("---")
    st.sidebar.metric("Total Data", f"{len(filtered_df):,}")
    if 'Stunting' in filtered_df.columns:
        # Hitung jumlah stunting (handle baik numerik maupun string)
        stunting_count = count_stunting(filtered_df['Stunting'])
        st.sidebar.metric("Data Stunting", f"{stunting_count:,}")
        if len(filtered_df) > 0:
            st.sidebar.metric("Persentase Stunting", f"{(stunting_count/len(filtered_df)*100):.2f}%")
        else:
            st.sidebar.metric("Persentase Stunting", "0.00%")
    
    # Informasi dataset yang digabung
    if 'Dataset_Source' in df.columns:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Dataset yang Digabung")
        dataset_counts = df['Dataset_Source'].value_counts()
        for source, count in dataset_counts.items():
            st.sidebar.text(f"{source}: {count:,} data")
    
    return page, filtered_df

