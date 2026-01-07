"""Halaman Data Explorer"""
import streamlit as st
import pandas as pd


def _prepare_dataframe_for_display(df):
    """Persiapkan DataFrame untuk ditampilkan dengan memastikan kompatibilitas Arrow"""
    df_display = df.copy()
    
    # Konversi kolom object ke string untuk kompatibilitas Arrow
    for col in df_display.columns:
        if df_display[col].dtype == 'object':
            # Coba konversi ke string, jika gagal biarkan seperti semula
            try:
                df_display[col] = df_display[col].astype(str)
            except:
                pass
    
    return df_display


def render_data_explorer(filtered_df):
    """Render halaman data explorer"""
    st.title("Data Explorer")
    st.markdown("---")
    
    st.subheader("Tabel Data")
    # Persiapkan DataFrame untuk kompatibilitas Arrow
    display_df = _prepare_dataframe_for_display(filtered_df)
    st.dataframe(display_df, width='stretch', height=400)
    
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Data sebagai CSV",
        data=csv,
        file_name='filtered_stunting_data_combined.csv',
        mime='text/csv'
    )
    
    st.markdown("---")
    st.subheader("Informasi Dataset")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Kolom Dataset:**")
        st.write(list(filtered_df.columns))
    
    with col2:
        st.write("**Tipe Data:**")
        st.write(filtered_df.dtypes)
    
    st.write("**Shape Dataset:**", filtered_df.shape)
    st.write("**Missing Values:**")
    st.write(filtered_df.isnull().sum())

