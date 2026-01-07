"""Halaman Prediksi"""
import streamlit as st
import plotly.express as px
from utils.model_utils import MODEL_AVAILABLE, load_model, preprocess_input, interpret_prediction
from constants import MODEL_PATH, COLORS


def render_prediction():
    """Render halaman prediksi"""
    st.title("Prediksi Stunting")
    st.markdown("---")
    
    if not MODEL_AVAILABLE:
        st.error("⚠️ TensorFlow/Keras tidak terinstall. Install dengan: `pip install tensorflow`")
        st.code("pip install tensorflow", language="bash")
        return
    
    # Tampilkan loading indicator
    with st.spinner("Memuat model..."):
        model = load_model(MODEL_PATH)
    
    if model is None:
        # Error sudah ditampilkan oleh load_model
        st.info("💡 **Tips:** Pastikan file model `best_stunting_model.h5` ada di folder yang sama dengan `dashboard.py`")
        return
    
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
        try:
            input_data = preprocess_input(
                sex_input, age_input, birth_weight, birth_length,
                body_weight, body_length, asi_input
            )
            
            # Validasi shape input
            if hasattr(model, 'input_shape') and model.input_shape:
                expected_features = model.input_shape[1] if len(model.input_shape) > 1 else model.input_shape[0]
                actual_features = input_data.shape[1]
                if expected_features != actual_features:
                    st.error(f"❌ **Error:** Jumlah fitur tidak sesuai! Model mengharapkan {expected_features} fitur, tetapi mendapat {actual_features} fitur.")
                    st.info("💡 Pastikan file `feature_scaler.pkl` sesuai dengan model yang digunakan.")
                    return
            
            # Cek apakah model adalah sklearn atau keras
            if hasattr(model, 'predict_proba'):
                # sklearn model
                prediction = model.predict_proba(input_data)
            else:
                # keras model
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
                st.plotly_chart(fig, width='stretch')
            
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
            import traceback
            st.error(f"**❌ Error saat melakukan prediksi:**")
            st.error(f"```\n{str(e)}\n```")
            with st.expander("🔍 Detail Error (untuk debugging)"):
                st.code(traceback.format_exc(), language="python")
            
            st.info("""
            **💡 Tips untuk mengatasi error:**
            - Pastikan semua input sudah diisi dengan benar
            - Pastikan model berhasil dimuat (lihat informasi model di atas)
            - Pastikan file `feature_scaler.pkl` ada dan valid
            - Cek apakah jumlah fitur sesuai dengan yang diharapkan model
            """)

