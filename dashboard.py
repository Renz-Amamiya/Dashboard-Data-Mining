import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# Import untuk model (opsional - hanya jika file model ada)
try:
    from tensorflow import keras
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

# Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard Analisis Stunting",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('dataset_stunting_balanced.csv')
    return df

df = load_data()

# Sidebar
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
sex_filter = st.sidebar.multiselect(
    "Jenis Kelamin",
    options=df['Sex'].unique(),
    default=df['Sex'].unique()
)

# Filter berdasarkan ASI Eksklusif
asi_filter = st.sidebar.multiselect(
    "ASI Eksklusif",
    options=df['ASI_Eksklusif'].unique(),
    default=df['ASI_Eksklusif'].unique()
)

# Filter berdasarkan Stunting
stunting_filter = st.sidebar.multiselect(
    "Status Stunting",
    options=df['Stunting'].unique(),
    default=df['Stunting'].unique()
)

# Filter umur
age_range = st.sidebar.slider(
    "Rentang Umur (bulan)",
    min_value=int(df['Age'].min()),
    max_value=int(df['Age'].max()),
    value=(int(df['Age'].min()), int(df['Age'].max()))
)

# Terapkan filter
filtered_df = df[
    (df['Sex'].isin(sex_filter)) &
    (df['ASI_Eksklusif'].isin(asi_filter)) &
    (df['Stunting'].isin(stunting_filter)) &
    (df['Age'] >= age_range[0]) &
    (df['Age'] <= age_range[1])
]

# Informasi di sidebar
st.sidebar.markdown("---")
st.sidebar.metric("Total Data", f"{len(filtered_df):,}")
st.sidebar.metric("Data Stunting", f"{filtered_df['Stunting'].sum():,}")
st.sidebar.metric("Persentase Stunting", f"{(filtered_df['Stunting'].sum()/len(filtered_df)*100):.2f}%")

# Halaman Overview
if page == "Overview":
    st.title("Dashboard Overview - Analisis Stunting")
    st.markdown("---")
    
    # Metrik utama
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Data",
            f"{len(filtered_df):,}",
            delta=f"{len(filtered_df) - len(df)}"
        )
    
    with col2:
        stunting_count = filtered_df['Stunting'].sum()
        st.metric(
            "Kasus Stunting",
            f"{stunting_count:,}",
            delta=f"{(stunting_count/len(filtered_df)*100) - (df['Stunting'].sum()/len(df)*100):.2f}%"
        )
    
    with col3:
        avg_age = filtered_df['Age'].mean()
        st.metric(
            "Rata-rata Umur",
            f"{avg_age:.1f} bulan",
            delta=f"{avg_age - df['Age'].mean():.1f}"
        )
    
    with col4:
        avg_weight = filtered_df['Body_Weight'].mean()
        st.metric(
            "Rata-rata Berat Badan",
            f"{avg_weight:.2f} kg",
            delta=f"{avg_weight - df['Body_Weight'].mean():.2f}"
        )
    
    st.markdown("---")
    
    # Grafik distribusi stunting
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribusi Status Stunting")
        stunting_counts = filtered_df['Stunting'].value_counts()
        fig_pie = px.pie(
            values=stunting_counts.values,
            names=['Tidak Stunting', 'Stunting'],
            color=['Tidak Stunting', 'Stunting'],
            color_discrete_map={'Tidak Stunting': '#2ecc71', 'Stunting': '#e74c3c'},
            hole=0.4
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(showlegend=True, height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("Distribusi berdasarkan Jenis Kelamin")
        sex_stunting = pd.crosstab(filtered_df['Sex'], filtered_df['Stunting'])
        sex_stunting_reset = sex_stunting.reset_index()
        sex_stunting_melted = pd.melt(
            sex_stunting_reset,
            id_vars=['Sex'],
            value_vars=[0, 1],
            var_name='Stunting',
            value_name='Jumlah'
        )
        sex_stunting_melted['Stunting_Label'] = sex_stunting_melted['Stunting'].map({0: 'Tidak Stunting', 1: 'Stunting'})
        fig_bar = px.bar(
            sex_stunting_melted,
            x='Sex',
            y='Jumlah',
            color='Stunting_Label',
            barmode='group',
            labels={'Sex': 'Jenis Kelamin', 'Jumlah': 'Jumlah'},
            color_discrete_map={'Tidak Stunting': '#2ecc71', 'Stunting': '#e74c3c'}
        )
        fig_bar.update_layout(
            xaxis_title="Jenis Kelamin",
            yaxis_title="Jumlah",
            legend_title="Status Stunting",
            height=400
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Grafik ASI Eksklusif
    st.subheader("Pengaruh ASI Eksklusif terhadap Stunting")
    asi_stunting = pd.crosstab(filtered_df['ASI_Eksklusif'], filtered_df['Stunting'])
    asi_stunting_reset = asi_stunting.reset_index()
    asi_stunting_melted = pd.melt(
        asi_stunting_reset,
        id_vars=['ASI_Eksklusif'],
        value_vars=[0, 1],
        var_name='Stunting',
        value_name='Jumlah'
    )
    asi_stunting_melted['Stunting_Label'] = asi_stunting_melted['Stunting'].map({0: 'Tidak Stunting', 1: 'Stunting'})
    fig_asi = px.bar(
        asi_stunting_melted,
        x='ASI_Eksklusif',
        y='Jumlah',
        color='Stunting_Label',
        barmode='group',
        labels={'ASI_Eksklusif': 'ASI Eksklusif', 'Jumlah': 'Jumlah'},
        color_discrete_map={'Tidak Stunting': '#2ecc71', 'Stunting': '#e74c3c'}
    )
    fig_asi.update_layout(
        xaxis_title="ASI Eksklusif",
        yaxis_title="Jumlah",
        legend_title="Status Stunting",
        height=400
    )
    st.plotly_chart(fig_asi, use_container_width=True)

# Halaman Analisis Visual
elif page == "Analisis Visual":
    st.title("Analisis Visual Data Stunting")
    st.markdown("---")
    
    # Pilihan visualisasi
    viz_type = st.selectbox(
        "Pilih Jenis Visualisasi",
        [
            "Distribusi Umur",
            "Hubungan Berat & Panjang Badan",
            "Hubungan Berat & Panjang Lahir",
            "Distribusi Berat Badan",
            "Distribusi Panjang Badan",
            "Heatmap Korelasi"
        ]
    )
    
    if viz_type == "Distribusi Umur":
        st.subheader("Distribusi Umur berdasarkan Status Stunting")
        fig = px.histogram(
            filtered_df,
            x='Age',
            color='Stunting',
            nbins=30,
            barmode='overlay',
            opacity=0.7,
            labels={'Age': 'Umur (bulan)', 'count': 'Frekuensi'},
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Hubungan Berat & Panjang Badan":
        st.subheader("Hubungan Berat Badan vs Panjang Badan")
        fig = px.scatter(
            filtered_df,
            x='Body_Length',
            y='Body_Weight',
            color='Stunting',
            size='Age',
            hover_data=['Sex', 'ASI_Eksklusif'],
            labels={
                'Body_Length': 'Panjang Badan (cm)',
                'Body_Weight': 'Berat Badan (kg)',
                'Stunting': 'Status Stunting'
            },
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Hubungan Berat & Panjang Lahir":
        st.subheader("Hubungan Berat Lahir vs Panjang Lahir")
        fig = px.scatter(
            filtered_df,
            x='Birth_Length',
            y='Birth_Weight',
            color='Stunting',
            size='Age',
            hover_data=['Sex', 'ASI_Eksklusif'],
            labels={
                'Birth_Length': 'Panjang Lahir (cm)',
                'Birth_Weight': 'Berat Lahir (kg)',
                'Stunting': 'Status Stunting'
            },
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Distribusi Berat Badan":
        st.subheader("Distribusi Berat Badan")
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.box(
                filtered_df,
                x='Stunting',
                y='Body_Weight',
                color='Stunting',
                labels={'Body_Weight': 'Berat Badan (kg)', 'Stunting': 'Status Stunting'},
                color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
            )
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.histogram(
                filtered_df,
                x='Body_Weight',
                color='Stunting',
                nbins=30,
                barmode='overlay',
                opacity=0.7,
                labels={'Body_Weight': 'Berat Badan (kg)', 'count': 'Frekuensi'},
                color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
    
    elif viz_type == "Distribusi Panjang Badan":
        st.subheader("Distribusi Panjang Badan")
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.box(
                filtered_df,
                x='Stunting',
                y='Body_Length',
                color='Stunting',
                labels={'Body_Length': 'Panjang Badan (cm)', 'Stunting': 'Status Stunting'},
                color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
            )
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.histogram(
                filtered_df,
                x='Body_Length',
                color='Stunting',
                nbins=30,
                barmode='overlay',
                opacity=0.7,
                labels={'Body_Length': 'Panjang Badan (cm)', 'count': 'Frekuensi'},
                color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
    
    elif viz_type == "Heatmap Korelasi":
        st.subheader("Heatmap Korelasi Variabel Numerik")
        numeric_cols = ['Age', 'Birth_Weight', 'Birth_Length', 'Body_Weight', 'Body_Length', 'Stunting']
        corr_matrix = filtered_df[numeric_cols].corr()
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu',
            labels=dict(x="Variabel", y="Variabel", color="Korelasi")
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

# Halaman Analisis Detail
elif page == "Analisis Detail":
    st.title("Analisis Detail Data Stunting")
    st.markdown("---")
    
    # Analisis berdasarkan kategori
    analysis_type = st.selectbox(
        "Pilih Analisis",
        [
            "Analisis berdasarkan Jenis Kelamin",
            "Analisis berdasarkan ASI Eksklusif",
            "Analisis berdasarkan Umur",
            "Statistik Deskriptif"
        ]
    )
    
    if analysis_type == "Analisis berdasarkan Jenis Kelamin":
        st.subheader("Analisis berdasarkan Jenis Kelamin")
        
        sex_analysis = filtered_df.groupby(['Sex', 'Stunting']).agg({
            'Age': 'mean',
            'Body_Weight': 'mean',
            'Body_Length': 'mean',
            'Birth_Weight': 'mean',
            'Birth_Length': 'mean'
        }).round(2)
        
        st.dataframe(sex_analysis, use_container_width=True)
        
        # Visualisasi
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Rata-rata Berat Badan', 'Rata-rata Panjang Badan', 
                          'Rata-rata Berat Lahir', 'Rata-rata Panjang Lahir'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        for i, (stunt_val, label) in enumerate([(0, 'Tidak Stunting'), (1, 'Stunting')]):
            temp_df = filtered_df[filtered_df['Stunting'] == stunt_val]
            sex_means = temp_df.groupby('Sex').agg({
                'Body_Weight': 'mean',
                'Body_Length': 'mean',
                'Birth_Weight': 'mean',
                'Birth_Length': 'mean'
            })
            
            fig.add_trace(
                go.Bar(x=sex_means.index, y=sex_means['Body_Weight'], name=f'Berat Badan - {label}'),
                row=1, col=1
            )
            fig.add_trace(
                go.Bar(x=sex_means.index, y=sex_means['Body_Length'], name=f'Panjang Badan - {label}'),
                row=1, col=2
            )
            fig.add_trace(
                go.Bar(x=sex_means.index, y=sex_means['Birth_Weight'], name=f'Berat Lahir - {label}'),
                row=2, col=1
            )
            fig.add_trace(
                go.Bar(x=sex_means.index, y=sex_means['Birth_Length'], name=f'Panjang Lahir - {label}'),
                row=2, col=2
            )
        
        fig.update_layout(height=700, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Analisis berdasarkan ASI Eksklusif":
        st.subheader("Analisis berdasarkan ASI Eksklusif")
        
        asi_analysis = filtered_df.groupby(['ASI_Eksklusif', 'Stunting']).agg({
            'Age': 'mean',
            'Body_Weight': 'mean',
            'Body_Length': 'mean',
            'Birth_Weight': 'mean',
            'Birth_Length': 'mean'
        }).round(2)
        
        st.dataframe(asi_analysis, use_container_width=True)
        
        # Persentase stunting berdasarkan ASI
        asi_stunt_pct = filtered_df.groupby('ASI_Eksklusif')['Stunting'].agg(['sum', 'count'])
        asi_stunt_pct['Persentase'] = (asi_stunt_pct['sum'] / asi_stunt_pct['count'] * 100).round(2)
        st.subheader("Persentase Stunting berdasarkan ASI Eksklusif")
        st.dataframe(asi_stunt_pct, use_container_width=True)
        
        # Visualisasi
        fig = px.bar(
            asi_stunt_pct.reset_index(),
            x='ASI_Eksklusif',
            y='Persentase',
            color='ASI_Eksklusif',
            labels={'Persentase': 'Persentase Stunting (%)'},
            color_discrete_sequence=['#3498db', '#e74c3c']
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Analisis berdasarkan Umur":
        st.subheader("Analisis berdasarkan Kelompok Umur")
        
        # Buat kelompok umur
        filtered_df['Kelompok_Umur'] = pd.cut(
            filtered_df['Age'],
            bins=[0, 12, 24, 36, 48, 60, 100],
            labels=['0-12 bulan', '13-24 bulan', '25-36 bulan', '37-48 bulan', '49-60 bulan', '>60 bulan']
        )
        
        age_analysis = filtered_df.groupby(['Kelompok_Umur', 'Stunting']).agg({
            'Body_Weight': 'mean',
            'Body_Length': 'mean'
        }).round(2)
        
        st.dataframe(age_analysis, use_container_width=True)
        
        # Visualisasi
        fig = px.bar(
            filtered_df.groupby(['Kelompok_Umur', 'Stunting']).size().reset_index(name='Jumlah'),
            x='Kelompok_Umur',
            y='Jumlah',
            color='Stunting',
            barmode='group',
            labels={'Jumlah': 'Jumlah Kasus'},
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
        )
        fig.update_layout(height=500, xaxis_title="Kelompok Umur", yaxis_title="Jumlah Kasus")
        st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Statistik Deskriptif":
        st.subheader("Statistik Deskriptif")
        
        numeric_cols = ['Age', 'Birth_Weight', 'Birth_Length', 'Body_Weight', 'Body_Length']
        desc_stats = filtered_df[numeric_cols].describe()
        st.dataframe(desc_stats, use_container_width=True)
        
        # Perbandingan stunting vs tidak stunting
        st.subheader("Perbandingan Statistik: Stunting vs Tidak Stunting")
        comparison = filtered_df.groupby('Stunting')[numeric_cols].agg(['mean', 'std', 'min', 'max']).round(2)
        st.dataframe(comparison, use_container_width=True)

# Halaman Data Explorer
elif page == "Data Explorer":
    st.title("Data Explorer")
    st.markdown("---")
    
    # Tampilkan data
    st.subheader("Tabel Data")
    st.dataframe(filtered_df, use_container_width=True, height=400)
    
    # Download data
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Data sebagai CSV",
        data=csv,
        file_name='filtered_stunting_data.csv',
        mime='text/csv'
    )
    
    # Informasi dataset
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

# Halaman Prediksi
elif page == "Prediksi":
    st.title("Prediksi Stunting")
    st.markdown("---")
    
    if not MODEL_AVAILABLE:
        st.warning("⚠️ TensorFlow/Keras tidak terinstall. Install dengan: pip install tensorflow")
        st.info("Untuk menggunakan fitur prediksi, pastikan file model H5 tersedia dan TensorFlow sudah terinstall.")
    else:
        # Cek apakah file model ada
        model_files = [f for f in os.listdir('.') if f.endswith('.h5') or f.endswith('.hdf5')]
        
        if not model_files:
            st.warning("⚠️ File model H5 tidak ditemukan di direktori saat ini.")
            st.info("""
            **Cara menggunakan fitur prediksi:**
            1. Pastikan file model H5 (misalnya: `model_terbaik.h5`) ada di folder yang sama dengan dashboard.py
            2. Model akan otomatis terdeteksi dan digunakan untuk prediksi
            """)
        else:
            # Load model
            @st.cache_resource
            def load_model(model_path):
                try:
                    model = keras.models.load_model(model_path)
                    return model
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
                    return None
            
            # Pilih model jika ada lebih dari satu
            if len(model_files) > 1:
                selected_model = st.selectbox("Pilih Model", model_files)
            else:
                selected_model = model_files[0]
            
            model = load_model(selected_model)
            
            if model is not None:
                st.success(f"Model berhasil dimuat: {selected_model}")
                
                # Tampilkan informasi model
                with st.expander("Informasi Model"):
                    try:
                        st.write(f"**Nama Model:** {selected_model}")
                        st.write(f"**Jumlah Layer:** {len(model.layers)}")
                        st.write(f"**Input Shape:** {model.input_shape}")
                        st.write(f"**Output Shape:** {model.output_shape}")
                        st.write(f"**Jumlah Parameter:** {model.count_params():,}")
                    except:
                        pass
                
                st.markdown("### Input Data untuk Prediksi")
                st.markdown("---")
                
                # Form input
                col1, col2 = st.columns(2)
                
                st.info("💡 **Tips:** Isi data dengan benar untuk mendapatkan prediksi yang akurat. Gunakan data aktual anak untuk hasil terbaik.")
                
                # Contoh nilai untuk testing - bisa diubah sesuai kebutuhan
                st.markdown("**Contoh Data untuk Testing:**")
                example_col1, example_col2, example_col3 = st.columns(3)
                with example_col1:
                    if st.button("Contoh: Anak Normal (24 bulan)", use_container_width=True):
                        st.session_state.test_age = 24
                        st.session_state.test_birth_weight = 3.2
                        st.session_state.test_birth_length = 48.5
                        st.session_state.test_body_weight = 12.5
                        st.session_state.test_body_length = 85.0
                        st.rerun()
                with example_col2:
                    if st.button("Contoh: Berisiko Stunting", use_container_width=True):
                        st.session_state.test_age = 30
                        st.session_state.test_birth_weight = 2.5
                        st.session_state.test_birth_length = 47.0
                        st.session_state.test_body_weight = 9.0
                        st.session_state.test_body_length = 75.0
                        st.rerun()
                with example_col3:
                    if st.button("Reset ke Default", use_container_width=True):
                        if 'test_age' in st.session_state:
                            del st.session_state.test_age
                        st.rerun()
                
                with col1:
                    sex_input = st.selectbox("Jenis Kelamin", ["Male", "Female"], help="Pilih jenis kelamin anak")
                    age_input = st.number_input("Umur (bulan)", min_value=0, max_value=120, 
                                                value=st.session_state.get('test_age', 24), 
                                                help="Umur anak dalam bulan (0-120 bulan)")
                    birth_weight = st.number_input("Berat Lahir (kg)", min_value=0.0, max_value=10.0, 
                                                  value=st.session_state.get('test_birth_weight', 3.2), step=0.1,
                                                  help="Berat badan saat lahir dalam kilogram")
                    birth_length = st.number_input("Panjang Lahir (cm)", min_value=0.0, max_value=100.0, 
                                                  value=st.session_state.get('test_birth_length', 48.5), step=0.1,
                                                  help="Panjang badan saat lahir dalam centimeter")
                
                with col2:
                    body_weight = st.number_input("Berat Badan Saat Ini (kg)", min_value=0.0, max_value=50.0, 
                                                  value=st.session_state.get('test_body_weight', 12.5), step=0.1,
                                                  help="Berat badan anak saat ini dalam kilogram")
                    body_length = st.number_input("Panjang Badan Saat Ini (cm)", min_value=0.0, max_value=150.0, 
                                                  value=st.session_state.get('test_body_length', 85.0), step=0.1,
                                                  help="Panjang badan anak saat ini dalam centimeter")
                    asi_input = st.selectbox("ASI Eksklusif", ["Yes", "No"], 
                                            help="Apakah anak mendapat ASI eksklusif?")
                
                st.markdown("---")
                
                # Fungsi preprocessing untuk mengubah input menjadi format yang diharapkan model
                def preprocess_input(sex, age, birth_weight, birth_length, body_weight, body_length, asi):
                    # Normalisasi berdasarkan range dari dataset (dari stunting_all_dl.csv)
                    # Range maksimum untuk normalisasi - disesuaikan dengan data training
                    # Dari analisis data: Age max ~60, tapi normalisasi menggunakan range yang lebih besar
                    age_max = 60.0  # Maksimum umur dalam bulan
                    birth_weight_max = 4.5  # Maksimum berat lahir (kg)
                    birth_length_max = 55.0  # Maksimum panjang lahir (cm)
                    body_weight_max = 20.0  # Maksimum berat badan (kg)
                    body_length_max = 110.0  # Maksimum panjang badan (cm)
                    
                    # Normalisasi fitur numerik (0-1 scaling) - sesuai dengan data training
                    age_norm = float(age) / age_max if age_max > 0 else 0.0
                    birth_weight_norm = float(birth_weight) / birth_weight_max if birth_weight_max > 0 else 0.0
                    birth_length_norm = float(birth_length) / birth_length_max if birth_length_max > 0 else 0.0
                    body_weight_norm = float(body_weight) / body_weight_max if body_weight_max > 0 else 0.0
                    body_length_norm = float(body_length) / body_length_max if body_length_max > 0 else 0.0
                    
                    # Age in years
                    age_years = float(age) / 12.0
                    
                    # BMI calculation
                    bmi = float(body_weight) / ((float(body_length) / 100.0) ** 2) if body_length > 0 else 0.0
                    
                    # Growth calculations
                    weight_growth = float(body_weight) - float(birth_weight)
                    length_growth = float(body_length) - float(birth_length)
                    
                    # Weight and Length per Age (normalized - berdasarkan data training)
                    # Dari data: weight_per_age max sekitar 1.1, length_per_age max sekitar 7.7
                    weight_per_age_raw = float(body_weight) / float(age) if age > 0 else 0.0
                    length_per_age_raw = float(body_length) / float(age) if age > 0 else 0.0
                    # Normalisasi berdasarkan range data training
                    weight_per_age = weight_per_age_raw / 1.2 if weight_per_age_raw > 0 else 0.0
                    length_per_age = length_per_age_raw / 8.0 if length_per_age_raw > 0 else 0.0
                    
                    # Binary features
                    low_birth_weight = 1.0 if birth_weight < 2.5 else 0.0
                    short_birth_length = 1.0 if birth_length < 48.0 else 0.0
                    
                    # ASI features
                    asi_numerical = 1.0 if asi == "Yes" else 0.0
                    asi_weight_growth = weight_growth if asi == "Yes" else 0.0
                    
                    # Nutritional stress (simplified calculation)
                    nutritional_stress = (1.0 - body_weight_norm) * (1.0 - body_length_norm)
                    
                    # Log transformations
                    log_body_weight = np.log(float(body_weight) + 1.0) if body_weight > 0 else 0.0
                    log_body_length = np.log(float(body_length) + 1.0) if body_length > 0 else 0.0
                    
                    # One-hot encoding for Sex
                    sex_female = 1.0 if sex == "Female" else 0.0
                    sex_male = 1.0 if sex == "Male" else 0.0
                    
                    # One-hot encoding for ASI
                    asi_no = 1.0 if asi == "No" else 0.0
                    asi_yes = 1.0 if asi == "Yes" else 0.0
                    
                    # Urutan fitur sesuai dengan model - 17 fitur pertama dari stunting_all_dl.csv
                    # Berdasarkan kolom: Age, Birth_Weight, Birth_Length, Body_Weight, Body_Length, 
                    # Age_Years, BMI, Weight_Growth, Length_Growth, Weight_per_Age, Length_per_Age,
                    # Low_Birth_Weight, Short_Birth_Length, ASI_Eksklusif_Numerical, ASI_Weight_Growth,
                    # Nutritional_Stress, Log_Body_Weight
                    features = [
                        age_norm,                    # 0: Age (normalized)
                        birth_weight_norm,           # 1: Birth_Weight (normalized)
                        birth_length_norm,           # 2: Birth_Length (normalized)
                        body_weight_norm,            # 3: Body_Weight (normalized)
                        body_length_norm,            # 4: Body_Length (normalized)
                        age_years,                   # 5: Age_Years
                        bmi,                         # 6: BMI
                        weight_growth,               # 7: Weight_Growth
                        length_growth,               # 8: Length_Growth
                        weight_per_age,              # 9: Weight_per_Age
                        length_per_age,              # 10: Length_per_Age
                        low_birth_weight,            # 11: Low_Birth_Weight
                        short_birth_length,          # 12: Short_Birth_Length
                        asi_numerical,               # 13: ASI_Eksklusif_Numerical
                        asi_weight_growth,           # 14: ASI_Weight_Growth
                        nutritional_stress,          # 15: Nutritional_Stress
                        log_body_weight              # 16: Log_Body_Weight (fitur ke-17)
                    ]
                    
                    # Pastikan ada 17 fitur
                    assert len(features) == 17, f"Expected 17 features, got {len(features)}"
                    
                    return np.array([features], dtype=np.float32)
                
                # Tombol prediksi
                if st.button("Prediksi Stunting", type="primary", use_container_width=True):
                    # Preprocess input data
                    input_data = preprocess_input(
                        sex_input,
                        age_input,
                        birth_weight,
                        birth_length,
                        body_weight,
                        body_length,
                        asi_input
                    )
                    
                    try:
                        # Debug: Tampilkan input data yang sudah di-preprocess
                        with st.expander("Debug: Input Data (17 fitur)", expanded=False):
                            feature_names = [
                                "Age (norm)", "Birth_Weight (norm)", "Birth_Length (norm)",
                                "Body_Weight (norm)", "Body_Length (norm)", "Age_Years",
                                "BMI", "Weight_Growth", "Length_Growth", "Weight_per_Age (norm)",
                                "Length_per_Age (norm)", "Low_Birth_Weight", "Short_Birth_Length",
                                "ASI_Eksklusif_Numerical", "ASI_Weight_Growth", "Nutritional_Stress",
                                "Log_Body_Weight"
                            ]
                            debug_df = pd.DataFrame({
                                'Feature': feature_names,
                                'Value': input_data[0]
                            })
                            st.dataframe(debug_df, use_container_width=True)
                        
                        # Prediksi
                        prediction = model.predict(input_data, verbose=0)
                        
                        # Debug: Tampilkan raw prediction
                        with st.expander("Debug: Raw Prediction", expanded=False):
                            st.write(f"Prediction shape: {prediction.shape}")
                            st.write(f"Prediction values: {prediction}")
                            st.write(f"Prediction type: {type(prediction[0][0])}")
                        
                        # Tampilkan hasil
                        st.markdown("### Hasil Prediksi")
                        st.markdown("---")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Interpretasi output model
                            # Jika output binary (sigmoid - 1 nilai: probabilitas stunting)
                            if len(prediction[0]) == 1:
                                prob_stunting = float(prediction[0][0])
                                # Pastikan probabilitas dalam range 0-1
                                prob_stunting = max(0.0, min(1.0, prob_stunting))
                                prob_no_stunting = 1.0 - prob_stunting
                                
                                # Threshold untuk klasifikasi
                                threshold = 0.5
                                if prob_stunting > threshold:
                                    result = "Stunting"
                                    result_color = "#e74c3c"
                                else:
                                    result = "Tidak Stunting"
                                    result_color = "#2ecc71"
                                
                                st.metric("Prediksi", result)
                                
                                # Visualisasi probabilitas
                                fig = px.bar(
                                    x=['Tidak Stunting', 'Stunting'],
                                    y=[prob_no_stunting, prob_stunting],
                                    color=['Tidak Stunting', 'Stunting'],
                                    color_discrete_map={'Tidak Stunting': '#2ecc71', 'Stunting': '#e74c3c'},
                                    labels={'x': 'Status', 'y': 'Probabilitas'},
                                    title='Probabilitas Prediksi'
                                )
                                fig.update_layout(height=400, showlegend=False)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Jika output multi-class atau softmax (2 nilai: [tidak_stunting, stunting])
                            else:
                                prob_no_stunting = float(prediction[0][0])
                                prob_stunting = float(prediction[0][1]) if len(prediction[0]) > 1 else 0.0
                                
                                # Normalisasi jika perlu (softmax sudah normalized)
                                total = prob_no_stunting + prob_stunting
                                if total > 0:
                                    prob_no_stunting = prob_no_stunting / total
                                    prob_stunting = prob_stunting / total
                                
                                # Threshold untuk klasifikasi
                                if prob_stunting > prob_no_stunting:
                                    result = "Stunting"
                                    result_color = "#e74c3c"
                                else:
                                    result = "Tidak Stunting"
                                    result_color = "#2ecc71"
                                
                                st.metric("Prediksi", result)
                                
                                # Visualisasi probabilitas
                                fig = px.bar(
                                    x=['Tidak Stunting', 'Stunting'],
                                    y=[prob_no_stunting, prob_stunting],
                                    color=['Tidak Stunting', 'Stunting'],
                                    color_discrete_map={'Tidak Stunting': '#2ecc71', 'Stunting': '#e74c3c'},
                                    labels={'x': 'Status', 'y': 'Probabilitas'},
                                    title='Probabilitas Prediksi'
                                )
                                fig.update_layout(height=400, showlegend=False)
                                st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.subheader("Detail Probabilitas")
                            st.metric("Tidak Stunting", f"{prob_no_stunting*100:.2f}%")
                            st.metric("Stunting", f"{prob_stunting*100:.2f}%")
                            
                            # Progress bar untuk probabilitas - konversi ke float Python
                            prob_stunting_float = float(prob_stunting)
                            st.progress(prob_stunting_float, text=f"Risiko Stunting: {prob_stunting*100:.1f}%")
                            
                            st.markdown("---")
                            st.subheader("Rekomendasi")
                            if result == "Stunting":
                                st.error("""
                                **Anak berisiko stunting. Rekomendasi:**
                                - Konsultasi dengan dokter spesialis anak
                                - Perbaikan gizi dan pola makan
                                - Monitoring pertumbuhan berkala
                                - Pastikan ASI eksklusif jika masih bayi
                                """)
                            else:
                                st.success("""
                                **Anak tidak berisiko stunting.**
                                - Tetap jaga pola makan dan gizi seimbang
                                - Lakukan monitoring rutin
                                - Pastikan asupan nutrisi tercukupi
                                """)
                    
                    except Exception as e:
                        st.error(f"Error saat melakukan prediksi: {str(e)}")
                        st.info("Pastikan struktur input data sesuai dengan yang diharapkan model.")

