import streamlit as st
import av
import numpy as np
import cv2
import time
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# ======================================================
# CONFIGURAÇÕES INICIAIS
# ======================================================

st.set_page_config(page_title="Monitor Cardíaco", layout="centered")
st.title("💡 Monitor Cardíaco em Tempo Real")

st.markdown("""
Este app capta o vídeo da câmera e calcula o **BPM (batimentos por minuto)** em tempo real.  
Posicione o **dedo sobre a câmera traseira com o flash ligado**.
""")

# Estado inicial
if "camera_on" not in st.session_state:
    st.session_state.camera_on = False
if "flash_on" not in st.session_state:
    st.session_state.flash_on = False

# ======================================================
# BOTÕES DE CONTROLE
# ======================================================

col1, col2 = st.columns(2)
with col1:
    st.session_state.camera_on = st.toggle("📸 Ligar/Desligar Câmera", value=st.session_state.camera_on)
with col2:
    st.session_state.flash_on = st.toggle("🔦 Ligar Flash", value=st.session_state.flash_on)

# ======================================================
# CONFIGURAÇÃO DO WEBRTC
# ======================================================

rtc_configuration = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

video_constraints = {
    "video": {
        "facingMode": {"ideal": "environment"},  # Usa câmera traseira
        "width": {"ideal": 1280},
        "height": {"ideal": 720},
        "frameRate": {"ideal": 30},  # FPS aumentado
        "torch": st.session_state.flash_on,
        "advanced": [{"torch": st.session_state.flash_on}],
    },
    "audio": False,
}

# ======================================================
# FUNÇÃO DE CÁLCULO DE BPM (exemplo base)
# ======================================================

class BPMProcessor:
    def __init__(self):
        self.buffer = []
        self.timestamps = []
        self.last_bpm = 0

    def process(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_val = np.mean(gray)
        self.buffer.append(mean_val)
        self.timestamps.append(time.time())

        if len(self.buffer) > 150:
            self.buffer.pop(0)
            self.timestamps.pop(0)
            signal = np.array(self.buffer)
            signal = (signal - np.mean(signal)) / np.std(signal)
            fft = np.fft.rfft(signal)
            freqs = np.fft.rfftfreq(len(signal), d=(self.timestamps[1] - self.timestamps[0]))
            bpm_range = (freqs * 60)
            bpm = bpm_range[np.argmax(np.abs(fft))]
            if 40 < bpm < 180:
                self.last_bpm = bpm

        cv2.putText(img, f"BPM: {int(self.last_bpm)}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

bpm_processor = BPMProcessor()

# ======================================================
# STREAMLIT WEBRTC
# ======================================================

if st.session_state.camera_on:
    webrtc_streamer(
        key="camera_bpm",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        media_stream_constraints=video_constraints,
        video_processor_factory=lambda: bpm_processor,
        async_processing=True,
    )

# ======================================================
# INTERFACE DE STATUS
# ======================================================

st.subheader("📊 BPM em tempo real")
if bpm_processor.last_bpm > 0:
    st.metric("Batimentos por Minuto (BPM)", f"{int(bpm_processor.last_bpm)}")
else:
    st.info("Aguardando cálculos de BPM… posicione o dedo e aguarde alguns segundos.")