# ==============================
# Importação das bibliotecas
# ==============================
import streamlit as st                    # Framework para criar a interface web
import cv2                                # OpenCV para processamento de imagem
import numpy as np                        # Numpy para cálculos numéricos
import threading                          # Para rodar cálculos em paralelo (sem travar vídeo)
import time                               # Controle de tempo
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase  # Captura de vídeo via WebRTC
import av                                 # Conversão de frames para exibição
from scipy.signal import butter, filtfilt  # Filtros digitais
from scipy.fft import rfft, rfftfreq       # FFT (Transformada Rápida de Fourier) para análise de frequência

# ==============================
# Configuração inicial do Streamlit
# ==============================
st.set_page_config(page_title="PPG Real-Time + Flash", layout="centered")
st.title("💡 Monitor Cardíaco em Tempo Real")  # Título da página

# CSS personalizado para esconder o status do WebRTC e rotacionar o vídeo (modo retrato)
st.markdown(
    """
    <style>
    div[data-testid="stWebRTCStatus"] { 
        display: none !important;  /* Esconde o status de conexão do WebRTC */
    }

    video {
        transform: rotate(90deg);  /* Rotaciona o vídeo 90 graus */
        object-fit: cover;         /* Faz o vídeo preencher toda a área */
        width: 100%;               /* Largura total */
        height: 80vh;              /* Altura proporcional à tela */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================
# Botão de ligar/desligar câmera
# ==============================
if "camera_on" not in st.session_state:
    st.session_state.camera_on = False  # Estado inicial da câmera (desligada)
st.toggle("Ligar/Desligar Câmera", key="camera_on")  # Botão de toggle na tela

# ==============================
# Série de BPM (histórico)
# ==============================
if "bpm_series" not in st.session_state:
    st.session_state.bpm_series = []  # Lista para armazenar valores de BPM ao longo do tempo
MAX_POINTS = 180  # Quantidade máxima de pontos no gráfico (~3 minutos)

# ==============================
# Classe de Processamento de Vídeo
# ==============================
class MeuProcessadorDeVideo(VideoProcessorBase):
    def __init__(self):
        # Buffers para armazenar as médias do canal vermelho ao longo do tempo
        self.buffer_R = []
        self.buffer_size = 150  # Tamanho do buffer (~5 segundos de dados)
        self.lock = threading.Lock()  # Controle de concorrência entre threads

        # Variáveis de estado
        self.bpm = 0.0          # Valor atual de BPM
        self.fps = 0.0          # Frames por segundo calculado
        self.frame_count = 0    # Contador de frames
        self.start_time = time.time()  # Tempo de início da contagem de FPS
        self.alpha_suavizacao = 0.2    # Fator de suavização para BPM
        self.dedo_detectado = False    # Flag indicando se o dedo está cobrindo a câmera
        self.RED_THRESHOLD = 220       # Limiar mínimo de intensidade vermelha para considerar dedo presente

        # Thread separada para calcular BPM continuamente
        self.calc_thread = threading.Thread(target=self.calcula_bpm_continuo, daemon=True)
        self.calc_thread.start()

    # -----------------------------
    # Filtro passa-faixa (Butterworth)
    # -----------------------------
    def butter_bandpass(self, lowcut, highcut, fs, order=3):
        nyq = 0.5 * fs                               # Frequência de Nyquist
        b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")  # Cria o filtro
        return b, a

    # -----------------------------
    # Aplica o filtro no sinal
    # -----------------------------
    def filtrar_sinal(self, data, lowcut, highcut, fs):
        b, a = self.butter_bandpass(lowcut, highcut, fs)
        return filtfilt(b, a, data)  # Aplica o filtro para frente e para trás (evita atraso de fase)

    # -----------------------------
    # Thread que calcula BPM continuamente
    # -----------------------------
    def calcula_bpm_continuo(self):
        while True:
            time.sleep(2.0)  # Espera 2 segundos entre os cálculos

            with self.lock:  # Evita acesso simultâneo ao buffer
                # Só calcula se há amostras suficientes e o dedo está detectado
                if not self.dedo_detectado or len(self.buffer_R) < self.buffer_size:
                    continue

                # Frequência de amostragem (frames por segundo)
                fps = self.fps if self.fps > 0 else 30.0

                # Normaliza o sinal (remove tendência e escala)
                data = np.array(self.buffer_R)
                data = (data - np.mean(data)) / (np.std(data) + 1e-9)

                # Filtra o sinal para isolar a faixa de frequência cardíaca
                filtered = self.filtrar_sinal(data, 0.7, 3.0, fps)

                # Calcula a FFT (Transformada Rápida de Fourier)
                freqs = rfftfreq(len(filtered), d=1.0 / fps)
                fft_vals = np.abs(rfft(filtered))

                # Filtra as frequências válidas (entre 0.7 e 3.0 Hz => 42 a 180 BPM)
                valid = (freqs >= 0.7) & (freqs <= 3.0)

                if np.any(valid):
                    # Encontra a frequência com maior energia (pico)
                    peak_freq = freqs[valid][np.argmax(fft_vals[valid])]
                    bpm = peak_freq * 60.0  # Converte Hz para BPM

                    # Mantém o valor apenas se for plausível
                    if 40 < bpm < 180:
                        # Suaviza a transição entre valores (filtro exponencial)
                        self.bpm = (
                            bpm if self.bpm == 0
                            else bpm * self.alpha_suavizacao + self.bpm * (1 - self.alpha_suavizacao)
                        )

    # -----------------------------
    # Função principal que processa cada frame de vídeo
    # -----------------------------
    def recv(self, frame):
        # Converte o frame do formato AV para um array OpenCV (BGR)
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape  # Altura e largura do frame

        # Define uma Região de Interesse (ROI) central (metade da tela)
        roi_size = int(min(h, w) * 0.5)
        x_i = (w - roi_size) // 2
        y_i = (h - roi_size) // 2
        roi = img[y_i:y_i + roi_size, x_i:x_i + roi_size]

        # -----------------------------
        # Cálculo de FPS (frames por segundo)
        # -----------------------------
        self.frame_count += 1
        now = time.time()
        if now - self.start_time >= 1.0:
            # Atualiza FPS a cada segundo
            self.fps = self.frame_count / (now - self.start_time)
            self.frame_count = 0
            self.start_time = now

        # -----------------------------
        # Detecção de dedo (com base na média do canal vermelho)
        # -----------------------------
        mean_red_value = np.mean(roi[:, :, 2]) if roi.size > 0 else 0
        self.dedo_detectado = mean_red_value > self.RED_THRESHOLD

        # -----------------------------
        # Atualiza o buffer com valores válidos
        # -----------------------------
        with self.lock:
            if self.dedo_detectado:
                # Adiciona o valor médio do vermelho ao buffer
                self.buffer_R.append(mean_red_value)
                if len(self.buffer_R) > self.buffer_size:
                    # Mantém o tamanho fixo do buffer
                    self.buffer_R.pop(0)
            else:
                # Se o dedo sai, limpa o buffer e zera o BPM
                self.buffer_R.clear()
                self.bpm = 0.0

        # -----------------------------
        # Mostra texto e retângulo na tela
        # -----------------------------
        if self.dedo_detectado and self.bpm > 0:
            texto = f"BPM: {self.bpm:.1f}"
            cor = (0, 255, 0)  # Verde
        elif self.dedo_detectado:
            texto = "Calculando BPM..."
            cor = (0, 255, 255)  # Amarelo
        else:
            texto = "Posicione o dedo"
            cor = (0, 0, 255)  # Vermelho

        # Mostra BPM, FPS e o ROI na tela
        cv2.putText(img, texto, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, cor, 3)
        cv2.putText(img, f"FPS: {self.fps:.1f}", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.rectangle(img, (x_i, y_i), (x_i + roi_size, y_i + roi_size), (0, 255, 0), 2)

        # Retorna o frame processado para exibição
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ==============================
# Configuração da câmera (WebRTC)
# ==============================
video_constraints = {
    "video": {
        "facingMode": {"ideal": "environment"},  # Usa a câmera traseira se disponível
        "width": {"ideal": 640},                 # Resolução ideal
        "height": {"ideal": 480},
        "frameRate": {"ideal": 30},              # 30 FPS para melhor precisão de sinal
        "torch": True,                           # Liga o flash (em celulares que suportam)
    },
    "audio": False,  # Desativa o áudio
}

# Cria o stream de vídeo usando WebRTC
ctx = webrtc_streamer(
    key="camera_flash",
    video_processor_factory=MeuProcessadorDeVideo,  # Classe que processa os frames
    media_stream_constraints=video_constraints,      # Configurações de câmera
    async_processing=True,                           # Processamento assíncrono
    desired_playing_state=st.session_state.camera_on, # Estado ligado/desligado
)

# ==============================
# Gráfico de BPM em tempo real
# ==============================
st.subheader("BPM em tempo real")
chart_placeholder = st.empty()  # Espaço reservado para o gráfico

# Atualiza  os  BPM se houver novos dados
if ctx and ctx.state.playing and ctx.video_processor:
    bpm_val = float(ctx.video_processor.bpm) if ctx.video_processor.bpm else 0.0
    if bpm_val > 0:
        st.session_state.bpm_series.append(bpm_val)
        if len(st.session_state.bpm_series) > MAX_POINTS:
            st.session_state.bpm_series = st.session_state.bpm_series[-MAX_POINTS:]

# mostra o gráfico ou mensagem de espera
if len(st.session_state.bpm_series) >= 2:
    chart_placeholder.line_chart(st.session_state.bpm_series)
else:
    chart_placeholder.info("Aguardando cálculos de BPM… posicione o dedo e aguarde alguns segundos.")

# ==============================
# Atualização periódica da interface
# ==============================
REFRESH_INTERVAL = 1.5  # Atualiza a cada 1.5s
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()
else:
    if time.time() - st.session_state.last_refresh > REFRESH_INTERVAL:
        st.session_state.last_refresh = time.time()
        # Força o recarregamento da interface (para atualizar o gráfico e BPM)
        if hasattr(st, "rerun"):
            st.rerun()
        elif hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
