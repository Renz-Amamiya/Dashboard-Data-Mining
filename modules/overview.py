"""Halaman Overview Dashboard"""
import streamlit as st
import pandas as pd
from utils.data_loader import create_crosstab_melted, count_stunting
from utils.visualizations import create_pie_chart, create_bar_chart


def render_overview(filtered_df):
    """Render halaman overview"""
    st.title("Dashboard Overview - Analisis Stunting")
    st.markdown("---")
    
    # Metrik utama
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Data", f"{len(filtered_df):,}")
    
    with col2:
        if 'Stunting' in filtered_df.columns:
            stunting_count = count_stunting(filtered_df['Stunting'])
            st.metric("Kasus Stunting", f"{stunting_count:,}")
        else:
            st.metric("Kasus Stunting", "N/A")
    
    with col3:
        if 'Age' in filtered_df.columns:
            avg_age = filtered_df['Age'].mean()
            st.metric("Rata-rata Umur", f"{avg_age:.1f} bulan")
        else:
            st.metric("Rata-rata Umur", "N/A")
    
    with col4:
        if 'Body_Weight' in filtered_df.columns:
            avg_weight = filtered_df['Body_Weight'].mean()
            st.metric("Rata-rata Berat Badan", f"{avg_weight:.2f} kg")
        else:
            st.metric("Rata-rata Berat Badan", "N/A")
    
    st.markdown("---")
    
    # Grafik distribusi stunting
    if 'Stunting' in filtered_df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribusi Status Stunting")
            stunting_counts = filtered_df['Stunting'].value_counts()
            
            # Map values ke label yang benar
            if pd.api.types.is_numeric_dtype(filtered_df['Stunting']):
                # Jika numerik, map 0 dan 1
                label_map = {0: 'Tidak Stunting', 1: 'Stunting'}
            else:
                # Jika string, map yes/no ke label
                positive_values = ['yes', 'stunting', '1', 'true', 'y']
                label_map = {
                    val: 'Stunting' if str(val).lower().strip() in positive_values else 'Tidak Stunting'
                    for val in stunting_counts.index
                }
            
            # Buat names dan values yang sesuai
            names = [label_map.get(val, str(val)) for val in stunting_counts.index]
            values = stunting_counts.values
            
            fig_pie = create_pie_chart(
                values,
                names,
                "Distribusi Status Stunting"
            )
            st.plotly_chart(fig_pie, width='stretch')
        
        with col2:
            if 'Sex' in filtered_df.columns:
                st.subheader("Distribusi berdasarkan Jenis Kelamin")
                sex_melted = create_crosstab_melted(filtered_df, 'Sex')
                fig_bar = create_bar_chart(
                    sex_melted,
                    'Sex',
                    'Jumlah',
                    'Stunting_Label',
                    "Distribusi berdasarkan Jenis Kelamin"
                )
                st.plotly_chart(fig_bar, width='stretch')
        
        # Grafik ASI Eksklusif
        if 'ASI_Eksklusif' in filtered_df.columns:
            st.subheader("Pengaruh ASI Eksklusif terhadap Stunting")
            asi_melted = create_crosstab_melted(filtered_df, 'ASI_Eksklusif')
            fig_asi = create_bar_chart(
                asi_melted,
                'ASI_Eksklusif',
                'Jumlah',
                'Stunting_Label',
                "Pengaruh ASI Eksklusif terhadap Stunting"
            )
            st.plotly_chart(fig_asi, width='stretch')

