import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# Import untuk model
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

# Konstanta
COLORS = {
    'no_stunting': '#2ecc71',
    'stunting': '#e74c3c',
    'asi_yes': '#3498db',
    'asi_no': '#e74c3c'
}

DATASETS = [
    'dataset_stunting_balanced.csv',
    'dataset_ml_train_processed.csv',
    'dataset_dl_test_processed.csv'
]

MODEL_PATH = 'best_stunting_model.h5'

# ==================== FUNGSI HELPER ====================

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
        df['Stunting_Label'] = df['Stunting'].map({0: 'Tidak Stunting', 1: 'Stunting'})
    return df

def create_crosstab_melted(df, index_col, value_col='Stunting'):
    """Helper untuk membuat crosstab yang sudah di-melt"""
    crosstab = pd.crosstab(df[index_col], df[value_col])
    crosstab_reset = crosstab.reset_index()
    melted = pd.melt(
        crosstab_reset,
        id_vars=[index_col],
        value_vars=[0, 1],
        var_name=value_col,
        value_name='Jumlah'
    )
    melted['Stunting_Label'] = melted[value_col].map({0: 'Tidak Stunting', 1: 'Stunting'})
    return melted

def create_bar_chart(df, x, y, color, title, height=400):
    """Helper untuk membuat bar chart"""
    fig = px.bar(
        df,
        x=x,
        y=y,
        color=color,
        barmode='group',
        color_discrete_map={'Tidak Stunting': COLORS['no_stunting'], 'Stunting': COLORS['stunting']},
        labels={'x': x, 'y': y}
    )
    fig.update_layout(
        title=title,
        height=height,
        showlegend=True
    )
    return fig

def create_pie_chart(values, names, title, height=400):
    """Helper untuk membuat pie chart"""
    fig = px.pie(
        values=values,
        names=names,
        color=names,
        color_discrete_map={'Tidak Stunting': COLORS['no_stunting'], 'Stunting': COLORS['stunting']},
        hole=0.4
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(title=title, height=height, showlegend=True)
    return fig

def create_histogram(df, x, color_col, title, height=500):
    """Helper untuk membuat histogram"""
    fig = px.histogram(
        df,
        x=x,
        color=color_col,
        nbins=30,
        barmode='overlay',
        opacity=0.7,
        color_discrete_map={0: COLORS['no_stunting'], 1: COLORS['stunting']}
    )
    fig.update_layout(title=title, height=height)
    return fig

def create_scatter(df, x, y, color_col, size_col, title, height=600):
    """Helper untuk membuat scatter plot"""
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color_col,
        size=size_col,
        hover_data=['Sex', 'ASI_Eksklusif'] if 'Sex' in df.columns else [],
        color_discrete_map={0: COLORS['no_stunting'], 1: COLORS['stunting']}
    )
    fig.update_layout(title=title, height=height)
    return fig

def create_box_plot(df, x, y, color_col, title, height=400):
    """Helper untuk membuat box plot"""
    fig = px.box(
        df,
        x=x,
        y=y,
        color=color_col,
        color_discrete_map={0: COLORS['no_stunting'], 1: COLORS['stunting']}
    )
    fig.update_layout(title=title, height=height)
    return fig

@st.cache_resource
def load_model(model_path):
    """Load model H5"""
    try:
        if os.path.exists(model_path):
            model = keras.models.load_model(model_path)
            return model
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_input(sex, age, birth_weight, birth_length, body_weight, body_length, asi):
    """Preprocess input untuk prediksi model"""
    # Normalisasi
    age_max = 60.0
    birth_weight_max = 4.5
    birth_length_max = 55.0
    body_weight_max = 20.0
    body_length_max = 110.0
    
    age_norm = float(age) / age_max if age_max > 0 else 0.0
    birth_weight_norm = float(birth_weight) / birth_weight_max if birth_weight_max > 0 else 0.0
    birth_length_norm = float(birth_length) / birth_length_max if birth_length_max > 0 else 0.0
    body_weight_norm = float(body_weight) / body_weight_max if body_weight_max > 0 else 0.0
    body_length_norm = float(body_length) / body_length_max if body_length_max > 0 else 0.0
    
    # Feature engineering
    age_years = float(age) / 12.0
    bmi = float(body_weight) / ((float(body_length) / 100.0) ** 2) if body_length > 0 else 0.0
    weight_growth = float(body_weight) - float(birth_weight)
    length_growth = float(body_length) - float(birth_length)
    weight_per_age = float(body_weight) / float(age) if age > 0 else 0.0
    length_per_age = float(body_length) / float(age) if age > 0 else 0.0
    
    # Binary features
    low_birth_weight = 1.0 if birth_weight < 2.5 else 0.0
    short_birth_length = 1.0 if birth_length < 48.0 else 0.0
    asi_numerical = 1.0 if asi == "Yes" else 0.0
    asi_weight_growth = weight_growth if asi == "Yes" else 0.0
    nutritional_stress = (1.0 - body_weight_norm) * (1.0 - body_length_norm)
    log_body_weight = np.log(float(body_weight) + 1.0) if body_weight > 0 else 0.0
    
    features = [
        age_norm, birth_weight_norm, birth_length_norm, body_weight_norm, body_length_norm,
        age_years, bmi, weight_growth, length_growth, weight_per_age, length_per_age,
        low_birth_weight, short_birth_length, asi_numerical, asi_weight_growth,
        nutritional_stress, log_body_weight
    ]
    
    return np.array([features], dtype=np.float32)

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

# ==================== SIDEBAR ====================

st.sidebar.title("Navigasi Dashboard")
st.sidebar.markdown("---")

# Load data (gabungkan semua dataset)
df = load_data()

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
    st.sidebar.metric("Data Stunting", f"{filtered_df['Stunting'].sum():,}")
    st.sidebar.metric("Persentase Stunting", f"{(filtered_df['Stunting'].sum()/len(filtered_df)*100):.2f}%")

# Informasi dataset yang digabung
if 'Dataset_Source' in df.columns:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Dataset yang Digabung")
    dataset_counts = df['Dataset_Source'].value_counts()
    for source, count in dataset_counts.items():
        st.sidebar.text(f"{source}: {count:,} data")

# ==================== HALAMAN OVERVIEW ====================

if page == "Overview":
    st.title("Dashboard Overview - Analisis Stunting")
    st.markdown("---")
    
    # Metrik utama
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Data", f"{len(filtered_df):,}")
    
    with col2:
        if 'Stunting' in filtered_df.columns:
        stunting_count = filtered_df['Stunting'].sum()
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
            fig_pie = create_pie_chart(
                stunting_counts.values,
                ['Tidak Stunting', 'Stunting'],
                "Distribusi Status Stunting"
            )
        st.plotly_chart(fig_pie, use_container_width=True)
    
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
        st.plotly_chart(fig_bar, use_container_width=True)
    
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
    st.plotly_chart(fig_asi, use_container_width=True)

# ==================== HALAMAN ANALISIS VISUAL ====================

elif page == "Analisis Visual":
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
    else:
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
            numeric_cols = [c for c in ['Age', 'Birth_Weight', 'Birth_Length', 'Body_Weight', 'Body_Length', 'Stunting'] if c in filtered_df.columns]
            if len(numeric_cols) >= 2:
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

# ==================== HALAMAN ANALISIS DETAIL ====================

elif page == "Analisis Detail":
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
    else:
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
        asi_stunt_pct = filtered_df.groupby('ASI_Eksklusif')['Stunting'].agg(['sum', 'count'])
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

# ==================== HALAMAN DATA EXPLORER ====================

elif page == "Data Explorer":
    st.title("Data Explorer")
    st.markdown("---")
    
    st.subheader("Tabel Data")
    st.dataframe(filtered_df, use_container_width=True, height=400)
    
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

# ==================== HALAMAN PREDIKSI ====================

elif page == "Prediksi":
    st.title("Prediksi Stunting")
    st.markdown("---")
    
    if not MODEL_AVAILABLE:
        st.warning("⚠️ TensorFlow/Keras tidak terinstall. Install dengan: pip install tensorflow")
    else:
        model = load_model(MODEL_PATH)
        
        if model is None:
            st.warning(f"⚠️ File model {MODEL_PATH} tidak ditemukan.")
        else:
            st.success(f"Model berhasil dimuat: {MODEL_PATH}")
            
                with st.expander("Informasi Model"):
                    try:
                    st.write(f"**Nama Model:** {MODEL_PATH}")
                        st.write(f"**Jumlah Layer:** {len(model.layers)}")
                        st.write(f"**Input Shape:** {model.input_shape}")
                        st.write(f"**Output Shape:** {model.output_shape}")
                        st.write(f"**Jumlah Parameter:** {model.count_params():,}")
                    except:
                        pass
                
                st.markdown("### Input Data untuk Prediksi")
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
            st.info("💡 **Tips:** Isi data dengan benar untuk mendapatkan prediksi yang akurat.")
                
            # Contoh data
                example_col1, example_col2, example_col3 = st.columns(3)
                with example_col1:
                if st.button("Contoh: Anak Normal", use_container_width=True):
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
                if st.button("Reset", use_container_width=True):
                        if 'test_age' in st.session_state:
                            del st.session_state.test_age
                        st.rerun()
                
                with col1:
                sex_input = st.selectbox("Jenis Kelamin", ["Male", "Female"])
                    age_input = st.number_input("Umur (bulan)", min_value=0, max_value=120, 
                                            value=st.session_state.get('test_age', 24))
                    birth_weight = st.number_input("Berat Lahir (kg)", min_value=0.0, max_value=10.0, 
                                              value=st.session_state.get('test_birth_weight', 3.2), step=0.1)
                    birth_length = st.number_input("Panjang Lahir (cm)", min_value=0.0, max_value=100.0, 
                                              value=st.session_state.get('test_birth_length', 48.5), step=0.1)
                
                with col2:
                    body_weight = st.number_input("Berat Badan Saat Ini (kg)", min_value=0.0, max_value=50.0, 
                                              value=st.session_state.get('test_body_weight', 12.5), step=0.1)
                    body_length = st.number_input("Panjang Badan Saat Ini (cm)", min_value=0.0, max_value=150.0, 
                                              value=st.session_state.get('test_body_length', 85.0), step=0.1)
                asi_input = st.selectbox("ASI Eksklusif", ["Yes", "No"])
                
                st.markdown("---")
                
                if st.button("Prediksi Stunting", type="primary", use_container_width=True):
                    input_data = preprocess_input(
                    sex_input, age_input, birth_weight, birth_length,
                    body_weight, body_length, asi_input
                    )
                    
                    try:
                        prediction = model.predict(input_data, verbose=0)
                    prob_no_stunting, prob_stunting, result = interpret_prediction(prediction)
                    
                        st.markdown("### Hasil Prediksi")
                        st.markdown("---")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                                st.metric("Prediksi", result)
                                fig = px.bar(
                                    x=['Tidak Stunting', 'Stunting'],
                                    y=[prob_no_stunting, prob_stunting],
                                    color=['Tidak Stunting', 'Stunting'],
                            color_discrete_map={'Tidak Stunting': COLORS['no_stunting'], 'Stunting': COLORS['stunting']},
                                    labels={'x': 'Status', 'y': 'Probabilitas'},
                                    title='Probabilitas Prediksi'
                                )
                                fig.update_layout(height=400, showlegend=False)
                                st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.subheader("Detail Probabilitas")
                            st.metric("Tidak Stunting", f"{prob_no_stunting*100:.2f}%")
                            st.metric("Stunting", f"{prob_stunting*100:.2f}%")
                        st.progress(float(prob_stunting), text=f"Risiko Stunting: {prob_stunting*100:.1f}%")
                            
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
