"""Halaman Analisis Visual"""
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.visualizations import create_histogram, create_scatter, create_box_plot


def render_visual_analysis(filtered_df):
    """Render halaman analisis visual"""
    st.title("Analisis Visual Data Stunting")
    st.markdown("---")
    
    viz_options = []
    if 'Age' in filtered_df.columns and 'Stunting' in filtered_df.columns:
        viz_options.append("Distribusi Umur")
    if 'Body_Weight' in filtered_df.columns and 'Body_Length' in filtered_df.columns:
        viz_options.append("Hubungan Berat & Panjang Badan")
    if 'Birth_Weight' in filtered_df.columns and 'Birth_Length' in filtered_df.columns:
        viz_options.append("Hubungan Berat & Panjang Lahir")
    if 'Body_Weight' in filtered_df.columns:
        viz_options.append("Distribusi Berat Badan")
    if 'Body_Length' in filtered_df.columns:
        viz_options.append("Distribusi Panjang Badan")
    if len([c for c in ['Age', 'Birth_Weight', 'Birth_Length', 'Body_Weight', 'Body_Length', 'Stunting'] if c in filtered_df.columns]) >= 3:
        viz_options.append("Heatmap Korelasi")
    
    if not viz_options:
        st.warning("Dataset yang dipilih tidak memiliki kolom yang cukup untuk visualisasi.")
        return
    
    viz_type = st.selectbox("Pilih Jenis Visualisasi", viz_options)
    
    if viz_type == "Distribusi Umur":
        st.subheader("Distribusi Umur berdasarkan Status Stunting")
        fig = create_histogram(filtered_df, 'Age', 'Stunting', "Distribusi Umur")
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Hubungan Berat & Panjang Badan":
        st.subheader("Hubungan Berat Badan vs Panjang Badan")
        fig = create_scatter(
            filtered_df,
            'Body_Length',
            'Body_Weight',
            'Stunting',
            'Age' if 'Age' in filtered_df.columns else None,
            "Hubungan Berat Badan vs Panjang Badan"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Hubungan Berat & Panjang Lahir":
        st.subheader("Hubungan Berat Lahir vs Panjang Lahir")
        fig = create_scatter(
            filtered_df,
            'Birth_Length',
            'Birth_Weight',
            'Stunting',
            'Age' if 'Age' in filtered_df.columns else None,
            "Hubungan Berat Lahir vs Panjang Lahir"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Distribusi Berat Badan":
        st.subheader("Distribusi Berat Badan")
        col1, col2 = st.columns(2)
        with col1:
            fig1 = create_box_plot(filtered_df, 'Stunting', 'Body_Weight', 'Stunting', "Box Plot Berat Badan")
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = create_histogram(filtered_df, 'Body_Weight', 'Stunting', "Histogram Berat Badan")
            st.plotly_chart(fig2, use_container_width=True)
    
    elif viz_type == "Distribusi Panjang Badan":
        st.subheader("Distribusi Panjang Badan")
        col1, col2 = st.columns(2)
        with col1:
            fig1 = create_box_plot(filtered_df, 'Stunting', 'Body_Length', 'Stunting', "Box Plot Panjang Badan")
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = create_histogram(filtered_df, 'Body_Length', 'Stunting', "Histogram Panjang Badan")
            st.plotly_chart(fig2, use_container_width=True)
    
    elif viz_type == "Heatmap Korelasi":
        st.subheader("Heatmap Korelasi Variabel Numerik")
        
        # Ambil kolom yang mungkin numeric
        potential_cols = ['Age', 'Birth_Weight', 'Birth_Length', 'Body_Weight', 'Body_Length']
        
        # Filter hanya kolom yang benar-benar numeric dan ada di dataframe
        numeric_cols = []
        for col in potential_cols:
            if col in filtered_df.columns:
                # Cek apakah kolom numeric
                if pd.api.types.is_numeric_dtype(filtered_df[col]):
                    # Coba konversi ke float untuk memastikan
                    try:
                        test_series = pd.to_numeric(filtered_df[col], errors='coerce')
                        if not test_series.isna().all():  # Pastikan ada nilai yang valid
                            numeric_cols.append(col)
                    except:
                        pass
        
        if len(numeric_cols) >= 2:
            try:
                # Ambil subset dataframe dengan kolom numeric saja
                df_numeric = filtered_df[numeric_cols].copy()
                
                # Pastikan semua kolom bisa dikonversi ke float
                for col in df_numeric.columns:
                    df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
                
                # Hapus baris dengan semua nilai NaN
                df_numeric = df_numeric.dropna(how='all')
                
                # Hapus kolom yang semua nilainya NaN
                df_numeric = df_numeric.dropna(axis=1, how='all')
                
                # Pastikan masih ada minimal 2 kolom
                if len(df_numeric.columns) >= 2 and len(df_numeric) > 0:
                    corr_matrix = df_numeric.corr()
                    fig = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale='RdBu',
                        labels=dict(x="Variabel", y="Variabel", color="Korelasi")
                    )
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Tidak cukup data numeric yang valid untuk membuat heatmap korelasi.")
            except Exception as e:
                st.error(f"Error saat menghitung korelasi: {str(e)}")
                st.info("Pastikan kolom-kolom numeric memiliki nilai yang valid.")
        else:
            st.warning("Tidak cukup kolom numeric untuk membuat heatmap korelasi (minimal 2 kolom).")

