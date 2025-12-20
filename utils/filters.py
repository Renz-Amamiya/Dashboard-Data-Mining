"""Fungsi untuk filter sidebar"""
import streamlit as st
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
        sex_filter = st.sidebar.multiselect(
            "Jenis Kelamin",
            options=df['Sex'].unique(),
            default=df['Sex'].unique()
        )
    else:
        sex_filter = []
    
    # Filter berdasarkan ASI Eksklusif
    if 'ASI_Eksklusif' in df.columns:
        asi_filter = st.sidebar.multiselect(
            "ASI Eksklusif",
            options=df['ASI_Eksklusif'].unique(),
            default=df['ASI_Eksklusif'].unique()
        )
    else:
        asi_filter = []
    
    # Filter berdasarkan Stunting
    if 'Stunting' in df.columns:
        stunting_filter = st.sidebar.multiselect(
            "Status Stunting",
            options=df['Stunting'].unique(),
            default=df['Stunting'].unique()
        )
    else:
        stunting_filter = []
    
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
    if 'Stunting' in df.columns:
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

