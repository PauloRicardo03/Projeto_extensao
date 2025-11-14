# ==============================
# Importação das bibliotecas
# ==============================
import streamlit as st #faz o app web e a interface
import cv2 # Serve pra processar imagem
import numpy as np #serve pra fazer calculos
import threading # faz o calculo do bpm rodar numa thread separada
import time # faz o calculo do tempo
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase #webrtc_streamer liga a camera do celular ao python, VideoProcessorBase passa oque fazer com cada frame que chega
import av # traduz o formato do video para o formato numpy e o cv2
from scipy.signal import butter, filtfilt, find_peaks #butter faz o filtro digital, filtfilt aplica o filtro pra limpar ruido, find_peaks pega os picos no sinal
from scipy.fft import rfft, rfftfreq # faz o calculo fft, rfft trtansforma o sinalem um grafico,rfftfreq gera as frequencias que o rfft usa 
import matplotlib.pyplot as plt
import tempfile #cria um local temporario pra armazenar o video
import os #apaga o arquivo temporario depois da analise


# Configuração do Streamlit

st.set_page_config(page_title="PPG Real-Time + Flash", layout="centered")#texto que aparece na aba do navegador
st.title("💡 Monitor Cardíaco em Tempo Real")#titulo na pagina web
# CSS personalizado
st.markdown(
    """
    <style>
    div[data-testid="stWebRTCStatus"] { 
        display: none !important;
    }
    video {
        transform: rotate(90deg);
        object-fit: cover;
        width: 100%;
        height: 80vh;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,#permitem o uso do css
)


# Seleção da fonte (Câmera ou Vídeo)

fonte = st.radio("Fonte do sinal", ("Câmera (WebRTC)",
                 "Vídeo (upload)"), horizontal=True)# botões pra escolher se quer em tempo real ou upload de video


# Série de BPM (histórico)

if "bpm_series" not in st.session_state:# ve se tem o bpm_series no st.session_state se não tiver ele cria ( isso acontece só na primeira vez qu abre)
    st.session_state.bpm_series = [] 
MAX_POINTS = 180  # ~3 min


# Funções utilitárias



def bandpass_filter(data, lowcut, highcut, fs):
   #data é o sinal bruto
   #lowcut é o menor sinal
   #highcut é o maior sinal
   #fs quantos frames por segundo

    if fs <= 2 * highcut:
        fs = 2 * highcut + 1.0  # Força um fs seguro

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    # Garante que os cortes estão entre 0 e 1
    low = max(0.01, min(0.99, low))
    high = max(0.01, min(0.99, high))

    b, a = butter(2, [low, high], btype="band")
    return filtfilt(b, a, data)


# ==============================
# MODO CÂMERA (WebRTC)
# ==============================
if fonte == "Câmera (WebRTC)":

    # Botão de ligar/desligar câmera
    TOGGLE_KEY = "camera_on_v5"
    if TOGGLE_KEY not in st.session_state:
        st.session_state[TOGGLE_KEY] = False
    camera_on = st.toggle("📷 Ligar/Desligar Câmera", key=TOGGLE_KEY)

    st.divider()

    # --- ÁREA DE INFORMAÇÕES (UI DO STREAMLIT) ---
    col_status, col_bpm = st.columns([2, 1])

    with col_status:
        status_display = st.empty()

    with col_bpm:
        bpm_display = st.empty()

    # ----------------------------------------------

    class MeuProcessadorDeVideo(VideoProcessorBase):
        def __init__(self):
            self.buffer_R = []
            self.buffer_size = 150
            self.lock = threading.Lock()
            self.bpm = 0.0
            self.fps = 0.0
            self.frame_count = 0
            self.start_time = time.time()
            self.dedo_detectado = False
            self.RED_THRESHOLD = 220
            self.calc_thread = threading.Thread(
                target=self.calcula_bpm_continuo, daemon=True)
            self.calc_thread.start()

        def calcula_bpm_continuo(self):
            while True:
                time.sleep(4.0)  # Recalcula a cada 4s
                with self.lock:
                    # Se não tiver dados suficientes, pula
                    if not self.dedo_detectado or len(self.buffer_R) < self.buffer_size:
                        continue

                    # --- CORREÇÃO DO ERRO AQUI ---
                    # Se o FPS real for muito baixo (< 7), usamos 30.0 para evitar o crash do filtro
                    fs_real = self.fps
                    fs = fs_real if fs_real > 7.0 else 30.0

                    try:
                        data = np.array(self.buffer_R)
                        data = (data - np.mean(data)) / (np.std(data) + 1e-9)

                        filtered = bandpass_filter(data, 0.7, 3.0, fs)

                        freqs = rfftfreq(len(filtered), d=1.0 / fs)
                        fft_vals = np.abs(rfft(filtered))

                        valid = (freqs >= 0.7) & (freqs <= 3.0)
                        if np.any(valid):
                            peak_freq = freqs[valid][np.argmax(
                                fft_vals[valid])]
                            bpm = peak_freq * 60.0
                            if 40 < bpm < 180:
                                self.bpm = bpm if self.bpm == 0 else bpm*0.2 + self.bpm*0.8
                    except Exception as e:
                        # Se der erro matemático, ignora este ciclo em vez de travar o programa
                        print(f"Erro no cálculo BPM: {e}")
                        continue

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            h, w, _ = img.shape
            roi_size = int(min(h, w) * 0.5)
            x_i = (w - roi_size) // 2
            y_i = (h - roi_size) // 2
            roi = img[y_i:y_i + roi_size, x_i:x_i + roi_size]

            # Cálculo FPS
            self.frame_count += 1
            now = time.time()
            if now - self.start_time >= 1.0:
                self.fps = self.frame_count / (now - self.start_time)
                self.frame_count = 0
                self.start_time = now

            mean_red_value = np.mean(roi[:, :, 2]) if roi.size > 0 else 0
            self.dedo_detectado = mean_red_value > self.RED_THRESHOLD

            with self.lock:
                if self.dedo_detectado:
                    self.buffer_R.append(mean_red_value)
                    if len(self.buffer_R) > self.buffer_size:
                        self.buffer_R.pop(0)
                else:
                    self.buffer_R.clear()
                    self.bpm = 0.0

            # --- FEEDBACK VISUAL APENAS DO QUADRADO ---
            if not self.dedo_detectado:
                cor_retangulo = (0, 0, 255)  # Vermelho
            elif self.bpm <= 0:
                cor_retangulo = (0, 255, 255)  # Amarelo
            else:
                cor_retangulo = (0, 255, 0)  # Verde

            cv2.rectangle(img, (x_i, y_i), (x_i + roi_size,
                          y_i + roi_size), cor_retangulo, 3)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    # Configuração WebRTC
    video_constraints = {
        "video": {
            "facingMode": {"ideal": "environment"},
            "width": {"ideal": 640},
            "height": {"ideal": 480},
            "frameRate": {"ideal": 30},
            "torch": True,
        },
        "audio": False,
    }

    ctx = webrtc_streamer(
        key="camera_flash",
        video_processor_factory=MeuProcessadorDeVideo,
        media_stream_constraints=video_constraints,
        async_processing=True,
        desired_playing_state=camera_on,
    )

    # ========= ATUALIZAÇÃO DA INTERFACE =========
    if ctx and ctx.state.playing and ctx.video_processor:
        vp = ctx.video_processor

        bpm_val = float(vp.bpm) if vp.bpm and vp.bpm > 0 else 0.0
        dedo_ok = getattr(vp, 'dedo_detectado', False)

        if not dedo_ok:
            status_display.warning(
                "👆 **Posicione o dedo** cobrindo a câmera e o flash.")
        elif bpm_val <= 0:
            status_display.info("⏳ **Calculando...** mantenha o dedo parado.")
        else:
            status_display.success("✅ **Leitura Estável!**")

        if bpm_val > 0:
            bpm_display.metric(label="BPM", value=f"{bpm_val:.0f}", delta="❤️")
            st.session_state.bpm_series.append(bpm_val)
            if len(st.session_state.bpm_series) > MAX_POINTS:
                st.session_state.bpm_series = st.session_state.bpm_series[-MAX_POINTS:]
        else:
            bpm_display.metric(label="BPM", value="--", delta_color="off")

    else:
        status_display.info("Aguardando inicialização da câmera...")
        bpm_display.metric(label="BPM", value="--")

    # ========= Gráfico =========
    st.subheader("Histórico")
    bpm_chart_placeholder = st.empty()

    if len(st.session_state.bpm_series) >= 1:
        bpm_chart_placeholder.line_chart(st.session_state.bpm_series)
    else:
        bpm_chart_placeholder.caption("O gráfico aparecerá aqui...")

    time.sleep(1.0)
    st.rerun()

# ==============================
# MODO VÍDEO (upload)
# ==============================
else:
    st.info("Envie um vídeo curto (ex.: .mp4) com o dedo cobrindo a câmera/lanterna para testar o BPM.")
    file = st.file_uploader("Selecionar vídeo", type=[
                            "mp4", "mov", "avi", "mkv"])
    aplicar_limiar = st.checkbox(
        "Aplicar limiar de dedo (vermelho alto)", value=False)
    iniciar = st.button("▶️ Analisar vídeo")

    bpm_video_placeholder = st.empty()
    debug_placeholder = st.empty()
    frame_preview = st.empty()

    if iniciar and file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
            tmp.write(file.read())
            temp_path = tmp.name

        cap = cv2.VideoCapture(temp_path)
        fps_file = cap.get(cv2.CAP_PROP_FPS) or 30.0
        roi_buffer_R = []
        bpm_display_video = 0.0
        alpha = 0.2
        RED_THRESHOLD = 10
        max_frames = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1200, 1200))
        bpm_series_video = []

        for i in range(max_frames):
            ok, frame = cap.read()
            if not ok:
                break

            small = cv2.resize(frame, (320, 240))
            frame_preview.image(cv2.cvtColor(
                small, cv2.COLOR_BGR2RGB), caption=f"Frame {i+1}")

            h, w, _ = frame.shape
            roi_size = int(min(h, w) * 0.5)
            x_i = (w - roi_size) // 2
            y_i = (h - roi_size) // 2
            roi = frame[y_i:y_i + roi_size, x_i:x_i + roi_size]

            mean_red = float(np.mean(roi[:, :, 2])) if roi.size > 0 else 0.0
            dedo_ok = (mean_red > RED_THRESHOLD) if aplicar_limiar else True

            if dedo_ok:
                roi_buffer_R.append(mean_red)
                if len(roi_buffer_R) > 150:
                    roi_buffer_R.pop(0)
                if len(roi_buffer_R) >= 60:
                    data = np.array(roi_buffer_R, dtype=np.float64)
                    data = (data - np.mean(data)) / (np.std(data) + 1e-9)
                    try:
                        # Proteção similar no modo vídeo
                        safe_fps = fps_file if fps_file > 6.0 else 30.0
                        filtered = bandpass_filter(data, 0.7, 3.0, safe_fps)
                        peaks, _ = find_peaks(
                            filtered, distance=int(0.3*safe_fps))
                        if len(peaks) >= 2:
                            rr = np.diff(peaks) / safe_fps
                            rr = rr[(rr > 0.3) & (rr < 2.0)]
                            if len(rr) > 0:
                                bpm_now = 60.0 / np.median(rr)
                                if 40 < bpm_now < 180:
                                    bpm_display_video = bpm_now if bpm_display_video == 0 else bpm_now * \
                                        alpha + bpm_display_video*(1-alpha)
                    except:
                        pass

                    if i % 10 == 0 and bpm_display_video > 0:
                        bpm_series_video.append(bpm_display_video)
                        bpm_video_placeholder.line_chart(bpm_series_video)
            time.sleep(0.01)
        cap.release()
        try:
            os.remove(temp_path)
        except:
            pass
