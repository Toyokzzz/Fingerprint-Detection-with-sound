import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from gtts import gTTS
import tempfile
import os
import base64
from PIL import Image
import time

# ===========================================
# SETUP
# ===========================================
# Hapus pygame karena tidak bekerja di Streamlit Cloud

# Set page configuration for SPA layout
st.set_page_config(
    page_title="Deteksi Angka Jari dengan Suara",
    page_icon="✋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Perbaiki path logo
logo_path = "IMG/LOGO UKM IT no Background.png"

# ===========================================
# SIDEBAR
# ===========================================
with st.sidebar:
    # Logo di sidebar
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    if os.path.exists(logo_path):
        try:
            logo = Image.open(logo_path)
            st.image(logo, use_container_width=True, output_format="PNG")
        except Exception as e:
            st.warning(f"⚠️ Gagal memuat logo: {e}")
    else:
        st.warning("⚠️ Logo tidak ditemukan")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Kontrol Kamera
    st.header("Kontrol Kamera")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📷 Aktifkan Kamera", type="primary", use_container_width=True):
            st.session_state.camera_active = True
            st.rerun()
    
    with col2:
        if st.button("⏹️ Matikan Kamera", use_container_width=True):
            st.session_state.camera_active = False
            st.rerun()
    
    # Copyright di paling bawah sidebar
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; margin-top: 20px;'>"
        "© 2025 UKM IT. Cipta Karya Informatika"
        "</div>", 
        unsafe_allow_html=True
    )

# ===========================================
# MAIN CONTENT AREA (SPA Layout)
# ===========================================
# Header utama
st.markdown("<h1 style='text-align: center;'>✋ Deteksi Angka Jari dengan Suara</h1>", unsafe_allow_html=True)

# ===========================================
# DETEKSI JARI
# ===========================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

angka_teks = {
    0: "nol", 1: "satu", 2: "dua", 3: "tiga", 4: "empat",
    5: "lima", 6: "enam", 7: "tujuh", 8: "delapan", 9: "sembilan", 10: "sepuluh"
}

def count_fingers(hand_landmarks, hand_label):
    finger_tips = [8, 12, 16, 20]
    finger_dips = [6, 10, 14, 18]
    fingers = []

    if hand_label == "Right":
        fingers.append(1 if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x else 0)
    else:
        fingers.append(1 if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x else 0)

    for tip, dip in zip(finger_tips, finger_dips):
        fingers.append(1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[dip].y else 0)

    return sum(fingers)

def play_audio(text):
    """Fungsi untuk memutar audio menggunakan HTML5 audio player"""
    try:
        # Generate audio file
        tts = gTTS(text=text, lang='id')
        audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        tts.save(audio_file.name)
        
        # Read audio file and encode to base64
        with open(audio_file.name, 'rb') as f:
            audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode()
        
        # Create HTML audio player
        audio_html = f'''
        <audio autoplay>
            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
        </audio>
        '''
        st.markdown(audio_html, unsafe_allow_html=True)
        
        # Clean up
        os.unlink(audio_file.name)
        
    except Exception as e:
        st.error(f"❌ Gagal memutar suara: {e}")

# Initialize session state
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'last_number' not in st.session_state:
    st.session_state.last_number = None

# Main content area
if st.session_state.camera_active:
    # Status indicator
    st.success("🟢 **Kamera Aktif** - Tunjukkan jari Anda ke kamera")
    
    # Camera feed container
    camera_container = st.container()
    
    with camera_container:
        try:
            cap = cv2.VideoCapture(0)
            with mp_hands.Hands(min_detection_confidence=0.7,
                                min_tracking_confidence=0.7,
                                max_num_hands=2) as hands:
                stframe = st.empty()
                st.session_state.last_number = None

                while st.session_state.camera_active:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("❌ Tidak dapat mengakses kamera")
                        st.session_state.camera_active = False
                        st.rerun()
                        break

                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = hands.process(rgb)

                    total_fingers = 0
                    if result.multi_hand_landmarks:
                        for hand_landmarks, hand_label in zip(result.multi_hand_landmarks, result.multi_handedness):
                            label = hand_label.classification[0].label
                            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                            total_fingers += count_fingers(hand_landmarks, label)

                    # Display finger count
                    cv2.putText(frame, f"Angka: {total_fingers}", (30, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                    
                    # Display frame
                    stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

                    # Text-to-speech when number changes
                    if total_fingers != st.session_state.last_number and total_fingers in angka_teks:
                        st.session_state.last_number = total_fingers
                        text_to_say = angka_teks[total_fingers]
                        
                        # Play audio using HTML5 audio player
                        play_audio(text_to_say)

            cap.release()
        except Exception as e:
            st.error(f"❌ Error akses kamera: {e}")
            st.session_state.camera_active = False
            st.rerun()
else:
    # Welcome state when camera is off
    st.info("🔴 **Kamera Tidak Aktif** - Klik 'Aktifkan Kamera' di sidebar untuk memulai")
    
    # Additional information or instructions can go here
    st.markdown("""
    ### 📋 Petunjuk Penggunaan:
    1. Klik tombol **📷 Aktifkan Kamera** di sidebar
    2. Tunggu hingga kamera menyala
    3. Tunjukkan jari tangan ke kamera
    4. Sistem akan mendeteksi jumlah jari dan mengucapkannya
    5. Klik **⏹️ Matikan Kamera** untuk menghentikan
    
    **Catatan:** Fitur suara menggunakan HTML5 Audio Player
    """)
    
    # Placeholder for camera area
    placeholder = st.empty()
    with placeholder.container():
        st.markdown(
            "<div style='border: 2px dashed #ccc; padding: 100px; text-align: center; border-radius: 10px;'>"
            "<h3>📷 Area Kamera</h3>"
            "<p>Kamera akan muncul di sini ketika diaktifkan</p>"
            "</div>",
            unsafe_allow_html=True
        )
