"""Halaman Analisis Detail"""
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.visualizations import create_bar_chart
from utils.data_loader import count_stunting
from constants import COLORS


def render_detail_analysis(filtered_df):
    """Render halaman analisis detail"""
    st.title("Analisis Detail Data Stunting")
    st.markdown("---")
    
    analysis_options = []
    if 'Sex' in filtered_df.columns and 'Stunting' in filtered_df.columns:
        analysis_options.append("Analisis berdasarkan Jenis Kelamin")
    if 'ASI_Eksklusif' in filtered_df.columns and 'Stunting' in filtered_df.columns:
        analysis_options.append("Analisis berdasarkan ASI Eksklusif")
    if 'Age' in filtered_df.columns:
        analysis_options.append("Analisis berdasarkan Umur")
    analysis_options.append("Statistik Deskriptif")
    
    if not analysis_options:
        st.warning("Dataset yang dipilih tidak memiliki kolom yang cukup untuk analisis detail.")
        return
    
    analysis_type = st.selectbox("Pilih Analisis", analysis_options)
    
    if analysis_type == "Analisis berdasarkan Jenis Kelamin":
        st.subheader("Analisis berdasarkan Jenis Kelamin")
        numeric_cols = [c for c in ['Age', 'Body_Weight', 'Body_Length', 'Birth_Weight', 'Birth_Length'] if c in filtered_df.columns]
        if numeric_cols:
            sex_analysis = filtered_df.groupby(['Sex', 'Stunting'])[numeric_cols].agg('mean').round(2)
            st.dataframe(sex_analysis, use_container_width=True)
    
    elif analysis_type == "Analisis berdasarkan ASI Eksklusif":
        st.subheader("Analisis berdasarkan ASI Eksklusif")
        numeric_cols = [c for c in ['Age', 'Body_Weight', 'Body_Length', 'Birth_Weight', 'Birth_Length'] if c in filtered_df.columns]
        if numeric_cols:
            asi_analysis = filtered_df.groupby(['ASI_Eksklusif', 'Stunting'])[numeric_cols].agg('mean').round(2)
            st.dataframe(asi_analysis, use_container_width=True)
        
        if 'Stunting' in filtered_df.columns:
            # Hitung jumlah stunting per kelompok ASI Eksklusif
            asi_stunt_pct = filtered_df.groupby('ASI_Eksklusif')['Stunting'].agg([
                lambda x: count_stunting(x), 'count'
            ])
            asi_stunt_pct.columns = ['sum', 'count']
            asi_stunt_pct['Persentase'] = (asi_stunt_pct['sum'] / asi_stunt_pct['count'] * 100).round(2)
            st.subheader("Persentase Stunting berdasarkan ASI Eksklusif")
            st.dataframe(asi_stunt_pct, use_container_width=True)
            
            fig = px.bar(
                asi_stunt_pct.reset_index(),
                x='ASI_Eksklusif',
                y='Persentase',
                color='ASI_Eksklusif',
                color_discrete_sequence=[COLORS['asi_yes'], COLORS['asi_no']]
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Analisis berdasarkan Umur":
        st.subheader("Analisis berdasarkan Kelompok Umur")
        if 'Age' in filtered_df.columns:
            filtered_df['Kelompok_Umur'] = pd.cut(
                filtered_df['Age'],
                bins=[0, 12, 24, 36, 48, 60, 100],
                labels=['0-12 bulan', '13-24 bulan', '25-36 bulan', '37-48 bulan', '49-60 bulan', '>60 bulan']
            )
            
            numeric_cols = [c for c in ['Body_Weight', 'Body_Length'] if c in filtered_df.columns]
            if numeric_cols:
                age_analysis = filtered_df.groupby(['Kelompok_Umur', 'Stunting'])[numeric_cols].agg('mean').round(2)
                st.dataframe(age_analysis, use_container_width=True)
            
            if 'Stunting' in filtered_df.columns:
                age_grouped = filtered_df.groupby(['Kelompok_Umur', 'Stunting']).size().reset_index(name='Jumlah')
                fig = create_bar_chart(
                    age_grouped,
                    'Kelompok_Umur',
                    'Jumlah',
                    'Stunting',
                    "Distribusi berdasarkan Kelompok Umur"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Statistik Deskriptif":
        st.subheader("Statistik Deskriptif")
        numeric_cols = [c for c in ['Age', 'Birth_Weight', 'Birth_Length', 'Body_Weight', 'Body_Length'] if c in filtered_df.columns]
        if numeric_cols:
            desc_stats = filtered_df[numeric_cols].describe()
            st.dataframe(desc_stats, use_container_width=True)
            
            if 'Stunting' in filtered_df.columns:
                st.subheader("Perbandingan Statistik: Stunting vs Tidak Stunting")
                comparison = filtered_df.groupby('Stunting')[numeric_cols].agg(['mean', 'std', 'min', 'max']).round(2)
                st.dataframe(comparison, use_container_width=True)

