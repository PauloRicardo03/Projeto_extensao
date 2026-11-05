# ==============================
# Importação das bibliotecas
# ==============================
from collections import deque  # Estrutura de fila circular rápida para a mediana móvel de estabilização.
import os                      # Usado para apagar o arquivo de vídeo temporário após a análise.
import tempfile                # Usado para criar um arquivo temporário para salvar o vídeo do upload.
import threading               # Para executar os cálculos de DSP em paralelo (thread) sem travar o vídeo.
import time                    # Para registrar timestamps reais de cada frame e controlar intervalos.

import av                      # Biblioteca que "traduz" os frames de vídeo entre o WebRTC e o NumPy/OpenCV.
import cv2                     # OpenCV, para processamento de imagem, recorte de ROI e detecção facial.
import numpy as np             # NumPy, para cálculos numéricos, interpolação e manipulação matricial.
import pandas as pd            # Pandas, usado para alimentar incrementalmente o gráfico sem piscar (add_rows).
import streamlit as st         # Biblioteca para criar o aplicativo web e a interface interativa.
import streamlit.components.v1 as components  # Usado para injetar o manifesto e Service Worker do PWA no navegador.
from scipy.fft import rfft, rfftfreq  # rfft/rfftfreq: Transformada Rápida de Fourier para sinais reais.
from scipy.signal import butter, detrend, filtfilt, find_peaks  # Filtros digitais, remoção de tendência e picos.
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

# Configuração de servidor STUN público do Google:
# Essencial para que o vídeo WebRTC funcione quando acessado pelo celular via túnel (Cloudflare/ngrok) ou redes diferentes.
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# ==============================
# Configuração inicial do Streamlit e PWA
# ==============================
st.set_page_config(page_title="PPG & rPPG Monitor Multiparamétrico", layout="centered")


def garantir_icones_pwa():
    """Gera automaticamente os ícones 192x192 e 512x512 na pasta static/ caso não existam."""
    raiz_projeto = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pasta_static = os.path.join(raiz_projeto, "static")
    os.makedirs(pasta_static, exist_ok=True)

    for tamanho in (192, 512):
        caminho_icone = os.path.join(pasta_static, f"icon-{tamanho}.png")
        if not os.path.exists(caminho_icone):
            img = np.zeros((tamanho, tamanho, 3), dtype=np.uint8)
            img[:] = (23, 17, 14)  # Fundo escuro (#0E1117 em BGR)
            centro = tamanho // 2
            raio = int(tamanho * 0.38)
            cv2.circle(img, (centro, centro), raio, (75, 75, 255), -1, lineType=cv2.LINE_AA)
            # Desenha linha de pulso (ECG) branca no centro do ícone
            pts = np.array(
                [
                    [int(tamanho * 0.22), centro],
                    [int(tamanho * 0.38), centro],
                    [int(tamanho * 0.44), int(tamanho * 0.30)],
                    [int(tamanho * 0.52), int(tamanho * 0.72)],
                    [int(tamanho * 0.60), int(tamanho * 0.42)],
                    [int(tamanho * 0.65), centro],
                    [int(tamanho * 0.78), centro],
                ],
                dtype=np.int32,
            )
            espessura = max(4, tamanho // 32)
            cv2.polylines(img, [pts], isClosed=False, color=(255, 255, 255), thickness=espessura, lineType=cv2.LINE_AA)
            cv2.imwrite(caminho_icone, img)


garantir_icones_pwa()

# Injeção do Manifesto PWA, Meta Tags Mobile (Android/iOS) e Service Worker no documento principal
components.html(
    """
    <script>
      (function() {
        const doc = window.parent ? window.parent.document : document;
        const nav = window.parent ? window.parent.navigator : navigator;

        if (!doc.querySelector('link[rel="manifest"]')) {
          const manifestLink = doc.createElement('link');
          manifestLink.rel = 'manifest';
          manifestLink.href = '/app/static/manifest.json';
          doc.head.appendChild(manifestLink);
        }

        if (!doc.querySelector('meta[name="theme-color"]')) {
          const themeMeta = doc.createElement('meta');
          themeMeta.name = 'theme-color';
          themeMeta.content = '#FF4B4B';
          doc.head.appendChild(themeMeta);
        }

        if (!doc.querySelector('meta[name="apple-mobile-web-app-capable"]')) {
          const appleCapable = doc.createElement('meta');
          appleCapable.name = 'apple-mobile-web-app-capable';
          appleCapable.content = 'yes';
          doc.head.appendChild(appleCapable);

          const appleIcon = doc.createElement('link');
          appleIcon.rel = 'apple-touch-icon';
          appleIcon.href = '/app/static/icon-192.png';
          doc.head.appendChild(appleIcon);
        }

        if ('serviceWorker' in nav) {
          nav.serviceWorker.register('/app/static/sw.js')
            .then(function(reg) {
              console.log('PWA Service Worker registrado:', reg.scope);
            })
            .catch(function(err) {
              console.warn('Falha ao registrar Service Worker:', err);
            });
        }
      })();
    </script>
    """,
    height=0,
    width=0,
)

st.title("💡 Monitor Multiparamétrico em Tempo Real (PPG & rPPG)")

# Injeta CSS personalizado para estilizar a página e impedir o efeito de "piscar" nas atualizações.
st.markdown(
    """
    <style>
    /* Esconde o texto de status "RUNNING" do componente WebRTC */
    div[data-testid="stWebRTCStatus"] { 
        display: none !important;
    }
    /* Impede que os componentes fiquem opacos ("piscando") durante atualizações em tempo real */
    div[data-stale="true"] {
        opacity: 1 !important;
        transition: none !important;
    }
    /* Estilização do player de vídeo */
    video {
        object-fit: cover;
        width: 100%;
        height: 65vh;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================
# Seleção da Fonte do Sinal
# ==============================
# Três modos de operação: Contato (Dedo), Remoto (Rosto/rPPG) e Offline (Upload de Vídeo).
fonte = st.radio(
    "Fonte do sinal",
    (
        "Dedo (Traseira + Flash)",
        "Rosto (Frontal - rPPG)",
        "Vídeo (Upload)",
    ),
    horizontal=True,
)


# ==============================
# Estabilizador Fisiológico e Funções de DSP
# ==============================
class EstabilizadorFisiologico:
    """
    Garante uma leitura segura e clinicamente plausível combinando:
    1. Filtro de Mediana Móvel (descarta picos isolados/outliers).
    2. Slew-Rate Limiter (limita a variação máxima por ciclo à fisiologia humana).
    3. Suavização Exponencial (EMA).
    """

    def __init__(self, janela_mediana: int = 7, alpha: float = 0.15, max_delta_por_ciclo: float = 1.2):
        self.historico = deque(maxlen=janela_mediana)
        self.valor_estavel = 0.0
        self.alpha = alpha
        self.max_delta = max_delta_por_ciclo

    def reset(self):
        """Limpa o histórico quando o dedo ou rosto sai do enquadramento."""
        self.historico.clear()
        self.valor_estavel = 0.0

    def atualizar(self, novo_valor: float) -> float:
        """Recebe uma nova estimativa bruta e retorna o valor estabilizado."""
        if novo_valor <= 0.0:
            return self.valor_estavel

        # Se já temos leituras anteriores, amortece desvios abruptos (> 25% da mediana atual)
        if len(self.historico) >= 3:
            mediana_atual = float(np.median(self.historico))
            if abs(novo_valor - mediana_atual) > 0.25 * mediana_atual:
                novo_valor = mediana_atual + np.sign(novo_valor - mediana_atual) * (0.10 * mediana_atual)

        self.historico.append(novo_valor)

        # Aguarda pelo menos 3 estimativas consistentes antes de exibir o primeiro valor
        if len(self.historico) < 3:
            return 0.0

        alvo = float(np.median(self.historico))

        if self.valor_estavel == 0.0:
            self.valor_estavel = alvo
        else:
            # Slew-Rate Limiter: impede que o número salte mais que 'max_delta' em um único ciclo
            delta = alvo - self.valor_estavel
            delta_limitado = float(np.clip(delta, -self.max_delta, self.max_delta))
            proposto = self.valor_estavel + delta_limitado
            # Média Móvel Exponencial (EMA)
            self.valor_estavel = self.alpha * proposto + (1.0 - self.alpha) * self.valor_estavel

        return self.valor_estavel


def reamostrar_uniforme(timestamps: np.ndarray, valores: np.ndarray, fs_alvo: float = 30.0) -> np.ndarray:
    """
    No WebRTC, os frames podem chegar com pequenas variações de tempo (jitter de rede).
    Esta função interpola os dados reais para uma grade de tempo perfeitamente uniforme (30 Hz),
    garantindo que o filtro Butterworth e a FFT operem sem distorção espectral.
    """
    if len(timestamps) < 10:
        return valores
    duracao = timestamps[-1] - timestamps[0]
    if duracao <= 1.0:
        return valores

    num_pontos = int(duracao * fs_alvo)
    if num_pontos < 16:
        return valores

    grade_tempo = np.linspace(timestamps[0], timestamps[-1], num_pontos)
    return np.interp(grade_tempo, timestamps, valores)


def bandpass_filter(
    data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 3
) -> np.ndarray:
    """
    Remove a tendência linear (detrend) e aplica um filtro passa-faixa Butterworth bidirecional (filtfilt),
    que preserva a fase do sinal fisiológico.
    """
    if fs <= 2.0 * highcut:
        fs = 2.0 * highcut + 1.0

    nyq = 0.5 * fs
    low = max(0.001, min(0.99, lowcut / nyq))
    high = max(0.001, min(0.99, highcut / nyq))

    if low >= high:
        low = high * 0.5

    b, a = butter(order, [low, high], btype="band")
    sinal_sem_tendencia = detrend(data)
    return filtfilt(b, a, sinal_sem_tendencia)


def estimar_frequencia_segura(
    signal_filtrado: np.ndarray,
    fs: float,
    f_min: float,
    f_max: float,
    snr_minimo: float = 1.6,
    usar_picos_tempo: bool = True,
) -> float:
    """
    Calcula a frequência dominante em Hz com validação de segurança:
    1. Aplica janela de Hann + Zero-Padding (n_fft=2048) para resolução fina (< 0.9 BPM).
    2. Avalia a Relação Sinal-Ruído (SNR): se não houver um pico claro (movimento), retorna 0.0.
    3. Combina a leitura da FFT com a mediana dos intervalos entre picos no tempo (find_peaks).
    """
    n = len(signal_filtrado)
    if n < 30:
        return 0.0

    # Janela de Hann reduz vazamento espectral nas bordas do buffer
    janela = np.hanning(n)
    sinal_janelado = signal_filtrado * janela

    # Zero-Padding para 2048 pontos evita que a FFT dê saltos em degraus grandes
    fft_size = max(n, 2048)
    freqs = rfftfreq(fft_size, d=1.0 / fs)
    fft_mag = np.abs(rfft(sinal_janelado, n=fft_size))

    valid = (freqs >= f_min) & (freqs <= f_max)
    if not np.any(valid):
        return 0.0

    mag_valid = fft_mag[valid]
    freqs_valid = freqs[valid]

    idx_max = int(np.argmax(mag_valid))
    pico_mag = mag_valid[idx_max]
    mediana_mag = float(np.median(mag_valid)) + 1e-9

    # Validação de Qualidade do Sinal (SQI / SNR)
    snr = pico_mag / mediana_mag
    if snr < snr_minimo:
        return 0.0

    freq_fft = float(freqs_valid[idx_max])

    if not usar_picos_tempo:
        return freq_fft

    # Consenso com o domínio do tempo (distância entre batimentos sistólicos)
    dist_min = max(1, int(fs / f_max * 0.75))
    peaks, _ = find_peaks(signal_filtrado, distance=dist_min, prominence=np.std(signal_filtrado) * 0.35)
    if len(peaks) >= 3:
        ibis = np.diff(peaks) / fs
        ibis_validos = ibis[(ibis >= 1.0 / f_max) & (ibis <= 1.0 / f_min)]
        if len(ibis_validos) >= 2:
            # Coeficiente de variação (CV): verifica se os batimentos estão regulares
            cv_ibi = np.std(ibis_validos) / (np.mean(ibis_validos) + 1e-9)
            if cv_ibi < 0.25:
                freq_tempo = 1.0 / float(np.median(ibis_validos))
                return 0.65 * freq_fft + 0.35 * freq_tempo

    return freq_fft


def calcular_spo2_empirico(buf_r: np.ndarray, buf_b: np.ndarray) -> float:
    """
    Calcula a saturação de oxigênio (SpO2 %) usando a Razão de Razões (Ros):
    - Componente DC: média estática de absorção luminosa (np.mean).
    - Componente AC: oscilação pulsátil sem tendência (np.std).
    - Fórmula empírica calibrada: SpO2 = 110.0 - 25.0 * Ros (limitado entre 70% e 100%).
    """
    dc_r = np.mean(buf_r)
    ac_r = np.std(detrend(buf_r))
    dc_b = np.mean(buf_b)
    ac_b = np.std(detrend(buf_b))

    if dc_r <= 1e-6 or dc_b <= 1e-6 or ac_b <= 1e-6:
        return 0.0

    r_os = (ac_r / (dc_r + 1e-9)) / (ac_b / (dc_b + 1e-9))
    r_os = float(np.clip(r_os, 0.4, 1.2))
    spo2 = 110.0 - (25.0 * r_os)
    return float(np.clip(spo2, 70.0, 100.0))


# ==============================================================================
# 1. MODO DEDO / TRASEIRA (PPG de Contato: BPM + SpO2)
# ==============================================================================
if fonte == "Dedo (Traseira + Flash)":

    TOGGLE_KEY = "camera_dedo_on"
    if TOGGLE_KEY not in st.session_state:
        st.session_state[TOGGLE_KEY] = False
    camera_on = st.toggle("📷 Ligar/Desligar Câmera Traseira", key=TOGGLE_KEY)

    st.divider()

    # Colunas para exibir o status e os cards de métricas (BPM e SpO2)
    col_status, col_bpm, col_spo2 = st.columns([2, 1, 1])
    with col_status:
        status_display = st.empty()
    with col_bpm:
        bpm_display = st.empty()
    with col_spo2:
        spo2_display = st.empty()

    class ProcessadorDedoPPG(VideoProcessorBase):
        def __init__(self):
            # Buffers sincronizados para os canais Vermelho (R), Azul (B) e timestamps
            self.buffer_R = []
            self.buffer_B = []
            self.timestamps = []
            self.buffer_size = 180   # ~6s de janela a 30 FPS
            self.min_samples = 75    # ~2.5s mínimos para iniciar o cálculo
            self.lock = threading.Lock()

            self.bpm = 0.0
            self.spo2 = 0.0
            self.estabilizador_bpm = EstabilizadorFisiologico(
                janela_mediana=7, alpha=0.15, max_delta_por_ciclo=1.2
            )
            self.estabilizador_spo2 = EstabilizadorFisiologico(
                janela_mediana=7, alpha=0.15, max_delta_por_ciclo=0.4
            )

            self.fps = 30.0
            self.frame_count = 0
            self.start_time = time.time()

            self.dedo_detectado = False
            self.RED_THRESHOLD = 175.0

            self.running = True
            self.calc_thread = threading.Thread(target=self._loop_calculo, daemon=True)
            self.calc_thread.start()

        def _loop_calculo(self):
            """Thread de processamento de sinais em segundo plano (roda a cada 0.5s)."""
            fs_uniforme = 30.0
            while self.running:
                time.sleep(0.5)
                with self.lock:
                    if not self.dedo_detectado or len(self.buffer_R) < self.min_samples:
                        continue
                    arr_r = np.array(self.buffer_R, dtype=np.float64)
                    arr_b = np.array(self.buffer_B, dtype=np.float64)
                    arr_t = np.array(self.timestamps, dtype=np.float64)

                try:
                    # 1. Reamostragem uniforme em 30 Hz
                    r_uniforme = reamostrar_uniforme(arr_t, arr_r, fs_alvo=fs_uniforme)

                    # Se houver variação abrupta de brilho (dedo escorregando), ignora a janela
                    if np.std(detrend(r_uniforme)) > 25.0:
                        continue

                    norm_r = (r_uniforme - np.mean(r_uniforme)) / (np.std(r_uniforme) + 1e-9)

                    # Filtra na banda cardíaca segura: 0.85 Hz (51 BPM) a 2.5 Hz (150 BPM)
                    filt_r = bandpass_filter(norm_r, 0.85, 2.5, fs_uniforme, order=3)
                    peak_hz = estimar_frequencia_segura(
                        filt_r, fs_uniforme, 0.85, 2.5, snr_minimo=1.6, usar_picos_tempo=True
                    )
                    bpm_bruto = peak_hz * 60.0

                    # 2. Cálculo de SpO2 (%) pelos canais R e B
                    spo2_bruto = calcular_spo2_empirico(arr_r, arr_b)

                    with self.lock:
                        if 50.0 <= bpm_bruto <= 150.0:
                            self.bpm = self.estabilizador_bpm.atualizar(bpm_bruto)
                        if 70.0 <= spo2_bruto <= 100.0:
                            self.spo2 = self.estabilizador_spo2.atualizar(spo2_bruto)
                except Exception as e:
                    print(f"[Erro DSP Dedo]: {e}")
                    continue

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            """Executado para cada frame recebido da câmera sem bloquear o vídeo."""
            img = frame.to_ndarray(format="bgr24")
            h, w, _ = img.shape

            # ROI Central (50% da menor dimensão)
            roi_size = int(min(h, w) * 0.5)
            x_i = (w - roi_size) // 2
            y_i = (h - roi_size) // 2
            roi = img[y_i : y_i + roi_size, x_i : x_i + roi_size]

            now = time.time()
            self.frame_count += 1
            elapsed = now - self.start_time
            if elapsed >= 1.0:
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                self.start_time = now

            if roi.size > 0:
                # Extrai simultaneamente a média dos canais Azul (0) e Vermelho (2)
                mean_bgr = np.mean(roi, axis=(0, 1))
                mean_b = float(mean_bgr[0])
                mean_r = float(mean_bgr[2])
            else:
                mean_b, mean_r = 0.0, 0.0

            dedo_ok = mean_r > self.RED_THRESHOLD

            with self.lock:
                self.dedo_detectado = dedo_ok
                if dedo_ok:
                    self.buffer_R.append(mean_r)
                    self.buffer_B.append(mean_b)
                    self.timestamps.append(now)
                    if len(self.buffer_R) > self.buffer_size:
                        self.buffer_R.pop(0)
                        self.buffer_B.pop(0)
                        self.timestamps.pop(0)
                else:
                    self.buffer_R.clear()
                    self.buffer_B.clear()
                    self.timestamps.clear()
                    self.estabilizador_bpm.reset()
                    self.estabilizador_spo2.reset()
                    self.bpm = 0.0
                    self.spo2 = 0.0

                bpm_atual = self.bpm

            # Cor do retângulo de feedback: Vermelho (sem dedo), Amarelo (calibrando), Verde (estável)
            if not dedo_ok:
                cor_retangulo = (0, 0, 255)
            elif bpm_atual <= 0:
                cor_retangulo = (0, 255, 255)
            else:
                cor_retangulo = (0, 255, 0)

            cv2.rectangle(img, (x_i, y_i), (x_i + roi_size, y_i + roi_size), cor_retangulo, 3)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    video_constraints_dedo = {
        "video": {
            "facingMode": {"ideal": "environment"},
            "width": {"ideal": 640},
            "height": {"ideal": 480},
            "frameRate": {"ideal": 30},
            "advanced": [{"torch": True}],
        },
        "audio": False,
    }

    ctx = webrtc_streamer(
        key="camera_dedo_ppg",
        video_processor_factory=ProcessadorDedoPPG,
        media_stream_constraints=video_constraints_dedo,
        rtc_configuration=RTC_CONFIGURATION,
        async_processing=True,
        desired_playing_state=camera_on,
    )

    st.subheader("Histórico em Tempo Real")
    # Instancia o gráfico uma única vez e adiciona linhas com add_rows() para não piscar a tela
    chart = st.line_chart(pd.DataFrame({"BPM": pd.Series(dtype=float)}))

    if ctx and ctx.state.playing:
        while ctx.state.playing:
            vp = ctx.video_processor
            if vp is None:
                time.sleep(0.2)
                continue

            with vp.lock:
                bpm_val = float(vp.bpm)
                spo2_val = float(vp.spo2)
                dedo_ok = bool(vp.dedo_detectado)
                progresso = min(100, int((len(vp.buffer_R) / vp.min_samples) * 100))

            if not dedo_ok:
                status_display.warning("👆 **Posicione o dedo** cobrindo a câmera traseira e o flash.")
            elif bpm_val <= 0:
                status_display.info(f"⏳ **Estabilizando leitura ({progresso}%)...** mantenha o dedo parado.")
            else:
                status_display.success("✅ **Leitura Estabilizada!**")

            if bpm_val > 0:
                bpm_display.metric(label="Frequência Cardíaca", value=f"{bpm_val:.0f} BPM", delta="❤️")
                chart.add_rows(pd.DataFrame({"BPM": [round(bpm_val, 1)]}))
            else:
                bpm_display.metric(label="Frequência Cardíaca", value="-- BPM")

            if spo2_val > 0:
                spo2_display.metric(label="Oximetria (SpO2)", value=f"{spo2_val:.1f} %", delta="🩸")
            else:
                spo2_display.metric(label="Oximetria (SpO2)", value="-- %")

            time.sleep(0.5)
    else:
        status_display.info("Aguardando inicialização da câmera traseira...")
        bpm_display.metric(label="Frequência Cardíaca", value="-- BPM")
        spo2_display.metric(label="Oximetria (SpO2)", value="-- %")


# ==============================================================================
# 2. MODO FACIAL / FRONTAL (rPPG: BPM + RPM)
# ==============================================================================
elif fonte == "Rosto (Frontal - rPPG)":

    TOGGLE_KEY_FACE = "camera_rosto_on"
    if TOGGLE_KEY_FACE not in st.session_state:
        st.session_state[TOGGLE_KEY_FACE] = False
    camera_face_on = st.toggle("🤳 Ligar/Desligar Câmera Frontal", key=TOGGLE_KEY_FACE)

    st.divider()

    col_status, col_bpm, col_rpm = st.columns([2, 1, 1])
    with col_status:
        status_display = st.empty()
    with col_bpm:
        bpm_display = st.empty()
    with col_rpm:
        rpm_display = st.empty()

    class ProcessadorRostoRPPG(VideoProcessorBase):
        def __init__(self):
            # Canal Verde (G) possui maior absorção pela oxi-hemoglobina sob luz ambiente visível
            self.buffer_G = []
            self.timestamps = []
            self.buffer_size = 300       # ~10s a 30 FPS
            self.min_samples_bpm = 105   # ~3.5s para estabilizar BPM facial
            self.min_samples_rpm = 160   # ~5.5s para estabilizar RPM
            self.lock = threading.Lock()

            self.bpm = 0.0
            self.rpm = 0.0
            self.estabilizador_bpm = EstabilizadorFisiologico(
                janela_mediana=7, alpha=0.15, max_delta_por_ciclo=1.2
            )
            self.estabilizador_rpm = EstabilizadorFisiologico(
                janela_mediana=7, alpha=0.15, max_delta_por_ciclo=0.5
            )

            self.fps = 30.0
            self.frame_count = 0
            self.total_frames = 0
            self.start_time = time.time()

            self.rosto_detectado = False
            self.last_face_box = None
            self.frames_sem_rosto = 0

            # Detector facial Haar Cascade nativo do OpenCV (leve e sem dependências externas pesadas)
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

            self.running = True
            self.calc_thread = threading.Thread(target=self._loop_calculo_rppg, daemon=True)
            self.calc_thread.start()

        def _loop_calculo_rppg(self):
            """Thread dedicada à extração de BPM e RPM via rPPG."""
            fs_uniforme = 30.0
            while self.running:
                time.sleep(0.5)
                with self.lock:
                    n_amostras = len(self.buffer_G)
                    if not self.rosto_detectado or n_amostras < self.min_samples_bpm:
                        continue
                    arr_g = np.array(self.buffer_G, dtype=np.float64)
                    arr_t = np.array(self.timestamps, dtype=np.float64)

                try:
                    g_uniforme = reamostrar_uniforme(arr_t, arr_g, fs_alvo=fs_uniforme)
                    norm_g = (g_uniforme - np.mean(g_uniforme)) / (np.std(g_uniforme) + 1e-9)

                    # 1. Filtro Cardíaco (BPM): 0.85 Hz a 2.4 Hz (51 a 144 BPM)
                    filt_cardiaco = bandpass_filter(norm_g, 0.85, 2.4, fs_uniforme, order=3)
                    peak_bpm_hz = estimar_frequencia_segura(
                        filt_cardiaco, fs_uniforme, 0.85, 2.4, snr_minimo=1.45, usar_picos_tempo=True
                    )
                    bpm_bruto = peak_bpm_hz * 60.0

                    # 2. Filtro Respiratório (RPM): 0.15 Hz a 0.4 Hz (9 a 24 respirações/min)
                    rpm_bruto = 0.0
                    if n_amostras >= self.min_samples_rpm:
                        filt_resp = bandpass_filter(norm_g, 0.15, 0.4, fs_uniforme, order=2)
                        peak_rpm_hz = estimar_frequencia_segura(
                            filt_resp, fs_uniforme, 0.15, 0.4, snr_minimo=1.3, usar_picos_tempo=False
                        )
                        rpm_bruto = peak_rpm_hz * 60.0

                    with self.lock:
                        if 50.0 <= bpm_bruto <= 145.0:
                            self.bpm = self.estabilizador_bpm.atualizar(bpm_bruto)
                        if 9.0 <= rpm_bruto <= 24.0:
                            self.rpm = self.estabilizador_rpm.atualizar(rpm_bruto)
                except Exception as e:
                    print(f"[Erro DSP rPPG]: {e}")
                    continue

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            h, w, _ = img.shape

            now = time.time()
            self.frame_count += 1
            self.total_frames += 1
            elapsed = now - self.start_time
            if elapsed >= 1.0:
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                self.start_time = now

            # Detecta o rosto em resolução reduzida (50%) a cada 2 frames para manter 30 FPS cravados
            if self.total_frames % 2 == 0 or self.last_face_box is None:
                small_gray = cv2.cvtColor(
                    cv2.resize(img, (w // 2, h // 2)), cv2.COLOR_BGR2GRAY
                )
                faces = self.face_cascade.detectMultiScale(
                    small_gray,
                    scaleFactor=1.2,
                    minNeighbors=5,
                    minSize=(40, 40),
                )

                if len(faces) > 0:
                    mx, my, mw, mh = max(faces, key=lambda r: r[2] * r[3])
                    novo_box = np.array([mx * 2, my * 2, mw * 2, mh * 2], dtype=np.float64)
                    if self.last_face_box is None:
                        self.last_face_box = novo_box
                    else:
                        # Suaviza a posição da bounding box para que a ROI da testa não trepide
                        self.last_face_box = 0.25 * novo_box + 0.75 * self.last_face_box
                    self.frames_sem_rosto = 0
                else:
                    self.frames_sem_rosto += 1
                    if self.frames_sem_rosto > 10:
                        self.last_face_box = None

            if self.last_face_box is not None:
                fx, fy, fw, fh = self.last_face_box.astype(int)

                # Isola a ROI da Testa (x + 0.3w, y + 0.1h, largura 0.4w, altura 0.2h)
                tx = max(0, int(fx + 0.3 * fw))
                ty = max(0, int(fy + 0.1 * fh))
                tw = int(0.4 * fw)
                th = int(0.2 * fh)

                tx_end = min(w, tx + tw)
                ty_end = min(h, ty + th)

                roi_testa = img[ty:ty_end, tx:tx_end]

                if roi_testa.size > 0:
                    # Extrai a média espacial do canal Verde (índice 1 no formato BGR)
                    mean_g = float(np.mean(roi_testa[:, :, 1]))

                    with self.lock:
                        self.rosto_detectado = True
                        self.buffer_G.append(mean_g)
                        self.timestamps.append(now)
                        if len(self.buffer_G) > self.buffer_size:
                            self.buffer_G.pop(0)
                            self.timestamps.pop(0)
                        bpm_atual = self.bpm
                else:
                    bpm_atual = 0.0

                cor_testa = (0, 255, 0) if bpm_atual > 0 else (0, 255, 255)
                cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), (255, 200, 0), 2)
                cv2.rectangle(img, (tx, ty), (tx_end, ty_end), cor_testa, 2)
            else:
                with self.lock:
                    self.rosto_detectado = False
                    self.buffer_G.clear()
                    self.timestamps.clear()
                    self.estabilizador_bpm.reset()
                    self.estabilizador_rpm.reset()
                    self.bpm = 0.0
                    self.rpm = 0.0

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    video_constraints_rosto = {
        "video": {
            "facingMode": {"ideal": "user"},
            "width": {"ideal": 640},
            "height": {"ideal": 480},
            "frameRate": {"ideal": 30},
            "torch": False,
        },
        "audio": False,
    }

    ctx_face = webrtc_streamer(
        key="camera_rosto_rppg",
        video_processor_factory=ProcessadorRostoRPPG,
        media_stream_constraints=video_constraints_rosto,
        rtc_configuration=RTC_CONFIGURATION,
        async_processing=True,
        desired_playing_state=camera_face_on,
    )

    st.subheader("Histórico em Tempo Real")
    chart_face = st.line_chart(pd.DataFrame({"BPM": pd.Series(dtype=float)}))

    if ctx_face and ctx_face.state.playing:
        while ctx_face.state.playing:
            vp = ctx_face.video_processor
            if vp is None:
                time.sleep(0.2)
                continue

            with vp.lock:
                bpm_val = float(vp.bpm)
                rpm_val = float(vp.rpm)
                rosto_ok = bool(vp.rosto_detectado)
                progresso = min(100, int((len(vp.buffer_G) / vp.min_samples_bpm) * 100))

            if not rosto_ok:
                status_display.warning("🙂 **Posicione seu rosto** de frente para a câmera em local iluminado.")
            elif bpm_val <= 0:
                status_display.info(f"⏳ **Estabilizando rPPG da testa ({progresso}%)...** evite movimentos.")
            else:
                status_display.success("✅ **Leitura rPPG Estabilizada!**")

            if bpm_val > 0:
                bpm_display.metric(label="Frequência Cardíaca", value=f"{bpm_val:.0f} BPM", delta="❤️")
                chart_face.add_rows(pd.DataFrame({"BPM": [round(bpm_val, 1)]}))
            else:
                bpm_display.metric(label="Frequência Cardíaca", value="-- BPM")

            if rpm_val > 0:
                rpm_display.metric(label="Freq. Respiratória", value=f"{rpm_val:.0f} RPM", delta="🫁")
            else:
                rpm_display.metric(label="Freq. Respiratória", value="-- RPM")

            time.sleep(0.5)
    else:
        status_display.info("Aguardando inicialização da câmera frontal...")
        bpm_display.metric(label="Frequência Cardíaca", value="-- BPM")
        rpm_display.metric(label="Freq. Respiratória", value="-- RPM")


# ==============================================================================
# 3. MODO VÍDEO (Upload)
# ==============================================================================
else:
    st.info("Envie um vídeo curto (ex.: .mp4) para extrair BPM e SpO2 offline.")
    file = st.file_uploader("Selecionar vídeo", type=["mp4", "mov", "avi", "mkv"])
    aplicar_limiar = st.checkbox("Aplicar limiar de dedo (vermelho alto)", value=False)
    iniciar = st.button("▶️ Analisar vídeo")

    col_v_bpm, col_v_spo2 = st.columns(2)
    metric_v_bpm = col_v_bpm.empty()
    metric_v_spo2 = col_v_spo2.empty()

    chart_video = st.line_chart(pd.DataFrame({"BPM": pd.Series(dtype=float)}))
    frame_preview = st.empty()

    if iniciar and file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
            tmp.write(file.read())
            temp_path = tmp.name

        cap = cv2.VideoCapture(temp_path)
        fps_file = cap.get(cv2.CAP_PROP_FPS) or 30.0
        roi_buffer_R = []
        roi_buffer_B = []
        estab_bpm_v = EstabilizadorFisiologico(janela_mediana=7, alpha=0.15, max_delta_por_ciclo=1.2)
        estab_spo2_v = EstabilizadorFisiologico(janela_mediana=7, alpha=0.15, max_delta_por_ciclo=0.4)
        bpm_display_video = 0.0
        spo2_display_video = 0.0
        RED_THRESHOLD = 10.0
        max_frames = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1200, 1200))

        for i in range(max_frames):
            ok, frame = cap.read()
            if not ok:
                break

            if i % 3 == 0:
                small = cv2.resize(frame, (320, 240))
                frame_preview.image(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), caption=f"Frame {i + 1}")

            h, w, _ = frame.shape
            roi_size = int(min(h, w) * 0.5)
            x_i = (w - roi_size) // 2
            y_i = (h - roi_size) // 2
            roi = frame[y_i : y_i + roi_size, x_i : x_i + roi_size]

            if roi.size > 0:
                mean_bgr = np.mean(roi, axis=(0, 1))
                mean_b = float(mean_bgr[0])
                mean_r = float(mean_bgr[2])
            else:
                mean_b, mean_r = 0.0, 0.0

            dedo_ok = (mean_r > RED_THRESHOLD) if aplicar_limiar else True

            if dedo_ok:
                roi_buffer_R.append(mean_r)
                roi_buffer_B.append(mean_b)
                if len(roi_buffer_R) > 180:
                    roi_buffer_R.pop(0)
                    roi_buffer_B.pop(0)

                if len(roi_buffer_R) >= 75 and i % 5 == 0:
                    arr_r = np.array(roi_buffer_R, dtype=np.float64)
                    arr_b = np.array(roi_buffer_B, dtype=np.float64)
                    norm_r = (arr_r - np.mean(arr_r)) / (np.std(arr_r) + 1e-9)
                    try:
                        safe_fps = fps_file if fps_file > 6.0 else 30.0
                        filtered = bandpass_filter(norm_r, 0.85, 2.5, safe_fps, order=3)
                        peak_hz = estimar_frequencia_segura(
                            filtered, safe_fps, 0.85, 2.5, snr_minimo=1.5, usar_picos_tempo=True
                        )
                        bpm_now = peak_hz * 60.0
                        if 50.0 <= bpm_now <= 150.0:
                            bpm_display_video = estab_bpm_v.atualizar(bpm_now)

                        spo2_now = calcular_spo2_empirico(arr_r, arr_b)
                        if 70.0 <= spo2_now <= 100.0:
                            spo2_display_video = estab_spo2_v.atualizar(spo2_now)
                    except Exception:
                        pass

                    if bpm_display_video > 0:
                        metric_v_bpm.metric("BPM (Vídeo)", f"{bpm_display_video:.0f} BPM")
                        metric_v_spo2.metric("SpO2 Estimado", f"{spo2_display_video:.1f} %")
                        chart_video.add_rows(pd.DataFrame({"BPM": [round(bpm_display_video, 1)]}))

            time.sleep(0.01)

        cap.release()
        try:
            os.remove(temp_path)
        except OSError:
            pass
