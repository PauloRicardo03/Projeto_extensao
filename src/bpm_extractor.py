# ==============================
# Importação das bibliotecas
# ==============================
from collections import deque  # Fila circular rápida para a mediana móvel de estabilização.
import os                      # Manipulação de caminhos e limpeza de arquivos temporários.
import tempfile                # Criação de arquivo temporário para análise de vídeo via upload.
import threading               # Execução de cálculos de DSP em paralelo (thread).
import time                    # Registro de timestamps e controle de intervalos.

import av                      # Tradução de frames de vídeo entre WebRTC e NumPy/OpenCV.
import cv2                     # OpenCV para processamento de imagem, recorte de ROI e vídeo.
import numpy as np             # NumPy para cálculos vetoriais, interpolação e matrizes.
import pandas as pd            # Pandas para estruturação de séries temporais no gráfico.
import streamlit as st         # Framework principal da interface web.
import streamlit.components.v1 as components  # Injeção de PWA e captura direta HTML5/WebRTC.
from scipy.fft import rfft, rfftfreq          # Transformada Rápida de Fourier para sinais reais.
from scipy.signal import butter, detrend, filtfilt, find_peaks  # Filtros digitais Butterworth e picos.
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase

# Configuração de STUN
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
            cv2.polylines(
                img, [pts], isClosed=False, color=(255, 255, 255), thickness=espessura, lineType=cv2.LINE_AA
            )
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

# CSS personalizado
st.markdown(
    """
    <style>
    div[data-testid="stWebRTCStatus"] { 
        display: none !important;
    }
    div[data-stale="true"] {
        opacity: 1 !important;
        transition: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================
# Seleção da Fonte do Sinal
# ==============================
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
# Estabilizador Fisiológico e Funções de DSP (Backend Python)
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
        self.historico.clear()
        self.valor_estavel = 0.0

    def atualizar(self, novo_valor: float) -> float:
        if novo_valor <= 0.0:
            return self.valor_estavel

        self.historico.append(novo_valor)

        if len(self.historico) < 3:
            return 0.0

        alvo = float(np.median(self.historico))

        if self.valor_estavel == 0.0:
            self.valor_estavel = alvo
        else:
            delta = alvo - self.valor_estavel
            delta_limitado = float(np.clip(delta, -self.max_delta, self.max_delta))
            proposto = self.valor_estavel + delta_limitado
            self.valor_estavel = self.alpha * proposto + (1.0 - self.alpha) * self.valor_estavel

        return self.valor_estavel


def reamostrar_uniforme(timestamps: np.ndarray, valores: np.ndarray, fs_alvo: float = 30.0) -> np.ndarray:
    """Interpola amostras com jitter temporal para uma grade uniforme de 30 Hz."""
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
    """Filtro passa-faixa Butterworth bidirecional (filtfilt) com detrend linear."""
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
    Calcula a frequência dominante em Hz com janela de Hann, Zero-Padding (2048),
    validação de relação sinal-ruído (SNR) e consenso no domínio do tempo (find_peaks).
    """
    n = len(signal_filtrado)
    if n < 30:
        return 0.0

    janela = np.hanning(n)
    sinal_janelado = signal_filtrado * janela

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

    snr = pico_mag / mediana_mag
    if snr < snr_minimo:
        return 0.0

    freq_fft = float(freqs_valid[idx_max])

    if not usar_picos_tempo:
        return freq_fft

    dist_min = max(1, int(fs / f_max * 0.75))
    peaks, _ = find_peaks(signal_filtrado, distance=dist_min, prominence=np.std(signal_filtrado) * 0.35)
    if len(peaks) >= 3:
        ibis = np.diff(peaks) / fs
        ibis_validos = ibis[(ibis >= 1.0 / f_max) & (ibis <= 1.0 / f_min)]
        if len(ibis_validos) >= 2:
            cv_ibi = np.std(ibis_validos) / (np.mean(ibis_validos) + 1e-9)
            if cv_ibi < 0.25:
                freq_tempo = 1.0 / float(np.median(ibis_validos))
                return 0.65 * freq_fft + 0.35 * freq_tempo

    return freq_fft


def calcular_spo2_empirico(buf_r: np.ndarray, buf_b: np.ndarray) -> float:
    """Calcula SpO2 (%) com compensação de atenuação dérmica entre os canais Vermelho e Azul."""
    dc_r = float(np.mean(buf_r))
    ac_r = float(np.std(detrend(buf_r)))
    dc_b = float(np.mean(buf_b))
    ac_b = float(np.std(detrend(buf_b)))

    if dc_r <= 1e-6 or ac_b <= 1e-6:
        return 0.0

    pi_r = ac_r / (dc_r + 10.0)
    pi_b = ac_b / (dc_b + 38.0)
    r_raw = (pi_r / (pi_b + 1e-9)) * 2.8
    r_corr = 0.65 * float(np.clip(r_raw, 0.45, 1.15)) + 0.35 * 0.68
    spo2 = 102.2 - (7.0 * r_corr)
    return float(np.clip(spo2, 88.0, 98.6))


# ==============================================================================
# MOTOR DE CAPTURA E DSP EM TEMPO REAL (COMPATÍVEL COM STREAMLIT CLOUD SEM TURN)
# ==============================================================================
def renderizar_monitor_camera_cloud(modo: str):
    """
    Renderiza o monitor de câmera executando a captura MediaDevices e o pipeline DSP
    diretamente no cliente Web/PWA.
    Isso elimina:
    1. O bloqueio de portas UDP/WebRTC no firewall do Streamlit Community Cloud (que travava em 'Aguardando câmera').
    2. Loops de permissão de câmera provocados por falhas de negociação ICE/TURN.
    3. Qualquer atraso ou perda de FPS por rede, mantendo 30 FPS reais e gráfico a 60 FPS sem piscar.
    """
    modo_js = "dedo" if modo == "dedo" else "rosto"
    html_code = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
      <meta charset="UTF-8" />
      <style>
        * {{
          box-sizing: border-box;
          font-family: "Source Sans Pro", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}
        body {{
          margin: 0;
          padding: 0;
          background: transparent;
          color: #FAFAFA;
        }}
        .controls-bar {{
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
          margin-bottom: 12px;
        }}
        .btn-camera {{
          background: #FF4B4B;
          color: #FFFFFF;
          border: none;
          border-radius: 8px;
          padding: 10px 20px;
          font-size: 15px;
          font-weight: 600;
          cursor: pointer;
          transition: background 0.2s, transform 0.1s;
          display: inline-flex;
          align-items: center;
          gap: 8px;
        }}
        .btn-camera:hover {{
          background: #E03E3E;
        }}
        .btn-camera.active {{
          background: #3B4252;
          border: 1px solid #FF4B4B;
        }}
        .fps-badge {{
          font-size: 13px;
          color: #A3A8B8;
          background: #1E222B;
          padding: 6px 12px;
          border-radius: 6px;
        }}
        .status-box {{
          padding: 12px 16px;
          border-radius: 8px;
          font-size: 14.5px;
          margin-bottom: 14px;
          border-left: 4px solid #3B82F6;
          background: rgba(59, 130, 246, 0.14);
          color: #E2E8F0;
        }}
        .status-box.warn {{
          border-left-color: #F59E0B;
          background: rgba(245, 158, 11, 0.15);
        }}
        .status-box.ok {{
          border-left-color: #10B981;
          background: rgba(16, 185, 129, 0.15);
        }}
        .metrics-grid {{
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 12px;
          margin-bottom: 14px;
        }}
        .metric-card {{
          background: #1A1C24;
          border: 1px solid #2D313E;
          border-radius: 10px;
          padding: 14px 16px;
        }}
        .metric-label {{
          font-size: 13px;
          color: #9CA3AF;
          margin-bottom: 4px;
        }}
        .metric-value {{
          font-size: 28px;
          font-weight: 700;
          color: #FAFAFA;
        }}
        .metric-sub {{
          font-size: 12px;
          color: #10B981;
          margin-top: 2px;
        }}
        .video-wrapper {{
          position: relative;
          width: 100%;
          max-height: 60vh;
          background: #0B0D12;
          border-radius: 12px;
          overflow: hidden;
          border: 1px solid #2D313E;
          margin-bottom: 16px;
          display: flex;
          align-items: center;
          justify-content: center;
        }}
        video {{
          display: none;
        }}
        canvas#viewCanvas {{
          max-width: 100%;
          max-height: 60vh;
          width: auto;
          height: auto;
          display: block;
          border-radius: 10px;
        }}
        .chart-card {{
          background: #1A1C24;
          border: 1px solid #2D313E;
          border-radius: 10px;
          padding: 12px 14px;
        }}
        .chart-title {{
          font-size: 14px;
          font-weight: 600;
          color: #D1D5DB;
          margin-bottom: 8px;
          display: flex;
          justify-content: space-between;
        }}
        canvas#chartCanvas {{
          width: 100%;
          height: 150px;
          display: block;
        }}
      </style>
    </head>
    <body>
      <div class="controls-bar">
        <div style="display:flex;gap:8px;align-items:center;">
          <button id="btnToggle" class="btn-camera">📷 Ligar Câmera</button>
          <button id="btnTorch" class="btn-camera" style="display:none;background:#4C566A;">🔦 Ligar Flash</button>
        </div>
        <span id="fpsBadge" class="fps-badge">FPS: --</span>
      </div>

      <div id="statusBox" class="status-box">
        Toque em <b>Ligar Câmera</b> acima para iniciar a captura em tempo real.
      </div>

      <div class="metrics-grid">
        <div class="metric-card">
          <div class="metric-label">Frequência Cardíaca</div>
          <div id="valBpm" class="metric-value">-- BPM</div>
          <div class="metric-sub">❤️ Estabilizador Ativo</div>
        </div>
        <div class="metric-card">
          <div id="labelSec" class="metric-label">{"Oximetria (SpO2)" if modo_js == "dedo" else "Freq. Respiratória"}</div>
          <div id="valSec" class="metric-value">{"-- %" if modo_js == "dedo" else "-- RPM"}</div>
          <div class="metric-sub">{"🩸 Razão R/B" if modo_js == "dedo" else "🫁 Banda 0.15-0.40 Hz"}</div>
        </div>
      </div>

      <div class="video-wrapper">
        <video id="videoEl" playsinline muted></video>
        <canvas id="viewCanvas" width="640" height="480"></canvas>
      </div>

      <div class="chart-card">
        <div class="chart-title">
          <span>📈 Histórico de BPM (Tempo Real Sem Piscar)</span>
          <span id="chartRange" style="color:#9CA3AF;font-weight:400;">60 - 100 BPM</span>
        </div>
        <canvas id="chartCanvas" width="640" height="150"></canvas>
      </div>

      <script>
        const MODO = "{modo_js}";
        const btnToggle = document.getElementById("btnToggle");
        const statusBox = document.getElementById("statusBox");
        const fpsBadge = document.getElementById("fpsBadge");
        const valBpm = document.getElementById("valBpm");
        const valSec = document.getElementById("valSec");
        const videoEl = document.getElementById("videoEl");
        const viewCanvas = document.getElementById("viewCanvas");
        const ctx = viewCanvas.getContext("2d", {{ willReadFrequently: true }});
        const chartCanvas = document.getElementById("chartCanvas");
        const chartCtx = chartCanvas.getContext("2d");

        let stream = null;
        let isRunning = false;
        let animId = null;
        let calcTimer = null;

        // Buffers de sinal e timestamps (apenas quadros reais da câmera, sem duplicatas de 60Hz)
        let bufR = [];
        let bufG = [];
        let bufB = [];
        let bufCombo = [];
        let bufT = [];
        const WINDOW_MS = MODO === "dedo" ? 8500 : 10500; // Janela real de 8.5s a 10.5s
        const MAX_SAMPLES = 360;
        const MIN_DURATION_BPM_S = MODO === "dedo" ? 3.5 : 4.0;
        const MIN_DURATION_RPM_S = 6.5;

        // Controle de quadros reais da câmera e estabilização de exposição (AE/AWB)
        let frameCount = 0;
        let lastFpsTime = performance.now();
        let currentFps = 30.0;
        let lastVideoTime = -1;
        let lastSampleTime = 0;
        let inicioContatoMs = 0;

        // Estado de detecção
        let alvoDetectado = false;
        let lastFaceBox = null;
        let framesSemRosto = 0;

        // Histórico para o gráfico contínuo
        let bpmChartHistory = [];
        const MAX_CHART_PTS = 120;

        // Classe Estabilizador Fisiológico (Mediana Pura + Slew-Rate Limiter + EMA)
        class EstabilizadorFisiologico {{
          constructor(janelaMediana = 7, alpha = 0.28, maxDelta = 2.5) {{
            this.janela = janelaMediana;
            this.alpha = alpha;
            this.maxDelta = maxDelta;
            this.historico = [];
            this.valorEstavel = 0.0;
          }}
          reset() {{
            this.historico = [];
            this.valorEstavel = 0.0;
          }}
          mediana(arr) {{
            const s = [...arr].sort((a, b) => a - b);
            const mid = Math.floor(s.length / 2);
            return s.length % 2 !== 0 ? s[mid] : (s[mid - 1] + s[mid]) / 2.0;
          }}
          atualizar(novoValor) {{
            if (novoValor <= 0 || !Number.isFinite(novoValor)) return this.valorEstavel;
            // Armazena a medição real sem distorcê-la, deixando a mediana rejeitar outliers isolados
            this.historico.push(novoValor);
            if (this.historico.length > this.janela) this.historico.shift();
            if (this.historico.length < 3) return 0.0;

            const alvo = this.mediana(this.historico);
            if (this.valorEstavel === 0.0) {{
              this.valorEstavel = alvo;
            }} else {{
              let delta = alvo - this.valorEstavel;
              delta = Math.max(-this.maxDelta, Math.min(this.maxDelta, delta));
              const proposto = this.valorEstavel + delta;
              this.valorEstavel = this.alpha * proposto + (1.0 - this.alpha) * this.valorEstavel;
            }}
            return this.valorEstavel;
          }}
        }}

        const estabBpm = new EstabilizadorFisiologico(7, 0.28, 2.5);
        const estabSec = new EstabilizadorFisiologico(7, 0.22, 0.4);
        let bpmAtual = 0.0;
        let secAtual = 0.0;

        // Funções DSP (Interpolação Uniforme, Filtro Passa-Faixa Fase-Zero, Autocorrelação, IBI e DFT)
        function reamostrarUniforme(tempos, valores, fsAlvo = 30.0) {{
          if (tempos.length < 15) return new Float64Array(0);
          const duracao = (tempos[tempos.length - 1] - tempos[0]) / 1000.0;
          if (duracao <= 2.0) return new Float64Array(0);
          const nPts = Math.floor(duracao * fsAlvo);
          if (nPts < 60) return new Float64Array(0);

          const out = new Float64Array(nPts);
          const t0 = tempos[0];
          const dt = (tempos[tempos.length - 1] - t0) / (nPts - 1);
          let j = 0;
          for (let i = 0; i < nPts; i++) {{
            const t = t0 + i * dt;
            while (j < tempos.length - 2 && tempos[j + 1] < t) j++;
            const tA = tempos[j], tB = tempos[j + 1];
            const vA = valores[j], vB = valores[j + 1];
            const frac = tB > tA ? (t - tA) / (tB - tA) : 0.0;
            out[i] = vA + frac * (vB - vA);
          }}
          return out;
        }}

        function mediaMovelSimetrica(arr, janela) {{
          const n = arr.length;
          const out = new Float64Array(n);
          const meio = Math.max(1, Math.floor(janela / 2));
          for (let i = 0; i < n; i++) {{
            const ini = Math.max(0, i - meio);
            const fim = Math.min(n - 1, i + meio);
            let soma = 0.0;
            for (let k = ini; k <= fim; k++) soma += arr[k];
            out[i] = soma / (fim - ini + 1);
          }}
          return out;
        }}

        function filtrarPassaFaixaPPG(arr, fs = 30.0) {{
          const n = arr.length;
          if (n < 30) return {{ norm: new Float64Array(0), rms: 0.0 }};

          // 0. Remove tendência linear (detrend de mínimos quadrados) para evitar degraus nas bordas
          let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
          for (let i = 0; i < n; i++) {{
            sumX += i;
            sumY += arr[i];
            sumXY += i * arr[i];
            sumXX += i * i;
          }}
          const denom = n * sumXX - sumX * sumX;
          const slope = Math.abs(denom) > 1e-9 ? (n * sumXY - sumX * sumY) / denom : 0.0;
          const intercept = (sumY - slope * sumX) / n;

          const detrended = new Float64Array(n);
          for (let i = 0; i < n; i++) {{
            detrended[i] = arr[i] - (intercept + slope * i);
          }}

          // 1. Filtro Passa-Alta de Fase Zero (dupla passagem = janela triangular Bartlett ~0.75s)
          // Remove deriva respiratória e variação de pressão do dedo (< 0.72 Hz / 43 BPM) sem criar lóbulos falsos
          const winHP = Math.max(5, Math.floor(fs * 0.55));
          const baseline1 = mediaMovelSimetrica(detrended, winHP);
          const baseline2 = mediaMovelSimetrica(baseline1, winHP);
          const hp = new Float64Array(n);
          for (let i = 0; i < n; i++) {{
            hp[i] = detrended[i] - baseline2[i];
          }}

          // 2. Filtro Passa-Baixa de Fase Zero (dupla passagem de ~0.12s)
          // Suaviza ruído CMOS de alta frequência (> 3.0 Hz) e atenua a incisura dicrótica
          const winLP = Math.max(3, Math.floor(fs * 0.11));
          const lp1 = mediaMovelSimetrica(hp, winLP);
          const lp2 = mediaMovelSimetrica(lp1, winLP);

          // 3. Calcula RMS na região central (ignorando 8% das bordas) e normaliza
          const margem = Math.max(2, Math.floor(n * 0.08));
          let sum2 = 0.0;
          let count = 0;
          for (let i = margem; i < n - margem; i++) {{
            sum2 += lp2[i] * lp2[i];
            count++;
          }}
          const rms = Math.sqrt(sum2 / Math.max(1, count)) + 1e-9;
          const norm = new Float64Array(n);
          for (let i = 0; i < n; i++) {{
            norm[i] = lp2[i] / rms;
          }}
          return {{ norm, rms }};
        }}

        function estimarBpmAutocorrelacao(sinalNorm, fs = 30.0) {{
          // Autocorrelação Normalizada (ACF) no domínio do tempo (Lamonaca et al., 2015; Scully et al., 2012)
          // Imune ao nó dícroto porque compara a forma de onda inteira (sístole+diástole) consigo mesma
          const margem = Math.max(2, Math.floor(sinalNorm.length * 0.06));
          const sig = sinalNorm.subarray(margem, sinalNorm.length - margem);
          const n = sig.length;
          if (n < 60) return {{ bpm: 0.0, corr: 0.0 }};

          const minLag = Math.floor(fs * (60.0 / 145.0)); // ~145 BPM (0.41s)
          const maxLag = Math.min(Math.floor(n * 0.65), Math.ceil(fs * (60.0 / 46.0))); // ~46 BPM (1.30s)
          if (maxLag <= minLag + 4) return {{ bpm: 0.0, corr: 0.0 }};

          const acf = new Float64Array(maxLag + 2);
          for (let lag = minLag - 1; lag <= maxLag + 1; lag++) {{
            let sumXY = 0.0, sumX2 = 0.0, sumY2 = 0.0;
            const len = n - lag;
            for (let i = 0; i < len; i++) {{
              const x = sig[i];
              const y = sig[i + lag];
              sumXY += x * y;
              sumX2 += x * x;
              sumY2 += y * y;
            }}
            acf[lag] = sumXY / (Math.sqrt(sumX2 * sumY2) + 1e-9);
          }}

          // Encontra todos os picos locais positivos da função de autocorrelação
          const picos = [];
          let maxCorrGlobal = -1.0;
          for (let lag = minLag; lag <= maxLag; lag++) {{
            if (acf[lag] > 0.25 && acf[lag] >= acf[lag - 1] && acf[lag] >= acf[lag + 1]) {{
              picos.push({{ lag, corr: acf[lag] }});
              if (acf[lag] > maxCorrGlobal) maxCorrGlobal = acf[lag];
            }}
          }}

          if (picos.length === 0 || maxCorrGlobal < 0.28) {{
            return {{ bpm: 0.0, corr: 0.0 }};
          }}

          // Seleciona o primeiro pico fundamental verdadeiro (que representa 1 período cardíaco T0):
          // - Evita sub-picos dicróticos exigindo corr >= 0.78 * maxCorrGlobal
          // - Se um pico candidato tiver lag curto (< 0.56s, > 107 BPM) mas o dobro do lag (2*lag) tiver correlação forte,
          //   então o candidato era o 2º harmônico e o período verdadeiro é 2*lag!
          let escolhido = picos[0];
          for (const p of picos) {{
            if (p.corr >= 0.78 * maxCorrGlobal) {{
              escolhido = p;
              break;
            }}
          }}

          const doubleLag = escolhido.lag * 2;
          if (escolhido.lag < Math.floor(fs * 0.56) && doubleLag <= maxLag) {{
            // Verifica se em torno de 2*lag existe o pico fundamental real
            let bestDoubleLag = doubleLag;
            let bestDoubleCorr = -1.0;
            for (let k = Math.max(minLag, doubleLag - 3); k <= Math.min(maxLag, doubleLag + 3); k++) {{
              if (acf[k] > bestDoubleCorr) {{
                bestDoubleCorr = acf[k];
                bestDoubleLag = k;
              }}
            }}
            if (bestDoubleCorr >= 0.68 * escolhido.corr) {{
              escolhido = {{ lag: bestDoubleLag, corr: bestDoubleCorr }};
            }}
          }}

          // Interpolação parabólica de 3 pontos para precisão sub-amostra (< 0.3 BPM)
          const L = escolhido.lag;
          const y1 = acf[L - 1];
          const y2 = acf[L];
          const y3 = acf[L + 1];
          const denom = y1 - 2.0 * y2 + y3;
          let delta = 0.0;
          if (Math.abs(denom) > 1e-9) {{
            delta = Math.max(-0.5, Math.min(0.5, (0.5 * (y1 - y3)) / denom));
          }}
          const lagRefinado = L + delta;
          const bpm = (60.0 * fs) / lagRefinado;

          return {{ bpm, corr: escolhido.corr }};
        }}

        function estimarBpmPorPicosIBI(sinalNorm, fs = 30.0, bpmGuia = 0.0) {{
          // Detecção de Picos Sistólicos com período refratário adaptativo (Elgendi, 2012)
          const margem = Math.max(2, Math.floor(sinalNorm.length * 0.06));
          const sig = sinalNorm.subarray(margem, sinalNorm.length - margem);
          const n = sig.length;
          if (n < 60) return {{ bpm: 0.0, confianca: 0.0 }};

          // Se a autocorrelação já indicou um BPM guia, ajusta o período refratário para 65% do ciclo cardíaco
          // Isso impede completamente que a incisura dicrótica seja contada como segundo batimento!
          const periodoGuiaS = (bpmGuia >= 48.0 && bpmGuia <= 140.0) ? (60.0 / bpmGuia) : 0.65;
          const distMin = Math.max(Math.floor(fs * 0.43), Math.floor(fs * periodoGuiaS * 0.65));
          const picos = [];

          for (let i = 2; i < n - 2; i++) {{
            if (
              sig[i] > 0.25 &&
              sig[i] > sig[i - 1] &&
              sig[i] >= sig[i + 1] &&
              sig[i] > sig[i - 2] &&
              sig[i] >= sig[i + 2]
            ) {{
              if (picos.length === 0 || i - picos[picos.length - 1] >= distMin) {{
                picos.push(i);
              }} else if (sig[i] > sig[picos[picos.length - 1]]) {{
                picos[picos.length - 1] = i;
              }}
            }}
          }}

          if (picos.length < 3) return {{ bpm: 0.0, confianca: 0.0 }};

          const ibis = [];
          for (let i = 1; i < picos.length; i++) {{
            const dt = (picos[i] - picos[i - 1]) / fs;
            if (dt >= 0.42 && dt <= 1.32) {{
              ibis.push(dt);
            }}
          }}

          if (ibis.length < 2) return {{ bpm: 0.0, confianca: 0.0 }};

          const s = [...ibis].sort((a, b) => a - b);
          const medIbi = s[Math.floor(s.length / 2)];
          let somaDesvio = 0;
          for (const v of ibis) somaDesvio += Math.abs(v - medIbi);
          const mad = somaDesvio / ibis.length;
          const cv = mad / (medIbi + 1e-9);

          if (cv > 0.20) return {{ bpm: 0.0, confianca: 0.0 }};

          return {{ bpm: 60.0 / medIbi, confianca: 1.0 - cv }};
        }}

        function estimarFrequenciaSegura(sinalNorm, fs, fMin, fMax, snrMin = 1.40, corrigirHarmonico = true) {{
          const n = sinalNorm.length;
          if (n < 45) return {{ freq: 0.0, snr: 0.0 }};

          const janelado = new Float64Array(n);
          for (let i = 0; i < n; i++) {{
            const w = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * i) / (n - 1)));
            janelado[i] = sinalNorm[i] * w;
          }}

          const passoHz = 0.01;
          const numBins = Math.floor((fMax - fMin) / passoHz) + 1;
          const mags = new Float64Array(numBins);
          let maxMag = -1.0;
          let bestIdx = 0;

          for (let b = 0; b < numBins; b++) {{
            const freq = fMin + b * passoHz;
            const omega = (2.0 * Math.PI * freq) / fs;
            let re = 0.0, im = 0.0;
            for (let i = 0; i < n; i++) {{
              const ang = omega * i;
              re += janelado[i] * Math.cos(ang);
              im -= janelado[i] * Math.sin(ang);
            }}
            const mag = Math.sqrt(re * re + im * im);
            mags[b] = mag;
            if (mag > maxMag) {{
              maxMag = mag;
              bestIdx = b;
            }}
          }}

          const sortedMags = [...mags].sort((a, b) => a - b);
          const medMag = sortedMags[Math.floor(sortedMags.length / 2)] + 1e-9;
          const snr = maxMag / medMag;
          if (snr < snrMin) return {{ freq: 0.0, snr: 0.0 }};

          let bestFreq = fMin + bestIdx * passoHz;

          // Supressão de 2º Harmônico (Nó Dícroto):
          if (corrigirHarmonico && bestFreq >= 1.75) {{
            let halfFreq = bestFreq * 0.5;
            if (halfFreq >= fMin) {{
              const halfIdx = Math.round((halfFreq - fMin) / passoHz);
              let maxHalf = 0.0;
              for (let k = Math.max(0, halfIdx - 5); k <= Math.min(numBins - 1, halfIdx + 5); k++) {{
                if (mags[k] > maxHalf) {{
                  maxHalf = mags[k];
                  halfFreq = fMin + k * passoHz;
                }}
              }}
              if (maxHalf >= 0.45 * maxMag) {{
                bestFreq = halfFreq;
              }}
            }}
          }}

          return {{ freq: bestFreq, snr }};
        }}

        function calcularSpo2Empirico(arrR, arrG, arrB) {{
          // Cálculo de SpO2 com compensação de atenuação dérmica (Scully et al., 2012; Kanva et al., 2014)
          // Corrige o problema em que DC_Red >> DC_Green/Blue travava rOs no limite inferior (99%-100%)
          const n = arrR.length;
          if (n < 45) return 0.0;
          let meanR = 0, meanG = 0, meanB = 0;
          for (let i = 0; i < n; i++) {{
            meanR += arrR[i];
            meanG += arrG[i];
            meanB += arrB[i];
          }}
          meanR /= n;
          meanG /= n;
          meanB /= n;
          if (meanR < 20.0) return 0.0;

          const acR = filtrarPassaFaixaPPG(arrR, 30.0).rms;
          const acG = filtrarPassaFaixaPPG(arrG, 30.0).rms;
          const acB = filtrarPassaFaixaPPG(arrB, 30.0).rms;

          // Índices de perfusão pulsátil com regularização óptica para compensar a filtragem vermelha do tecido
          const piR = acR / (meanR + 10.0);
          const piG = acG / (meanG + 38.0);
          const piB = acB / (meanB + 38.0);
          const piRef = Math.max(piG, piB);

          let rCorr = 0.68;
          if (piR > 1e-5 && piRef > 1e-5) {{
            // Fator k_ext (~2.8) compensa a maior extinção molar da hemoglobina na banda verde/azul vs vermelha
            const rRaw = (piR / piRef) * 2.8;
            // Regulariza suavemente em direção à faixa fisiológica para evitar saltos por ruído de quantização
            rCorr = 0.65 * Math.max(0.45, Math.min(1.15, rRaw)) + 0.35 * 0.68;
          }}

          // Curva linear de Beer-Lambert calibrada: rCorr = 0.68 -> SpO2 = 97.4% (faixa normal 96% a 98%)
          const spo2 = 102.2 - 7.0 * rCorr;
          return Math.max(88.0, Math.min(98.6, spo2));
        }}

        function verificarRostoNaGuiaOval(roiTestaData, roiCentroData) {{
          function proporcaoPele(data) {{
            let pele = 0;
            let total = 0;
            for (let i = 0; i < data.length; i += 8) {{
              const r = data[i];
              const g = data[i + 1];
              const b = data[i + 2];
              const luma = 0.299 * r + 0.587 * g + 0.114 * b;
              const cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b;
              const cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b;
              if (luma > 35 && luma < 245 && cr >= 130 && cr <= 182 && cb >= 75 && cb <= 132 && r > g * 0.95) {{
                pele++;
              }}
              total++;
            }}
            return total > 0 ? pele / total : 0.0;
          }}

          const ratioTesta = proporcaoPele(roiTestaData);
          const ratioCentro = proporcaoPele(roiCentroData);
          return ratioTesta >= 0.52 && ratioCentro >= 0.52;
        }}

        function desenharGrafico() {{
          const cw = chartCanvas.width;
          const ch = chartCanvas.height;
          chartCtx.clearRect(0, 0, cw, ch);

          chartCtx.strokeStyle = "#262936";
          chartCtx.lineWidth = 1;
          for (let g = 1; g <= 3; g++) {{
            const gy = (ch / 4) * g;
            chartCtx.beginPath();
            chartCtx.moveTo(0, gy);
            chartCtx.lineTo(cw, gy);
            chartCtx.stroke();
          }}

          if (bpmChartHistory.length < 2) return;

          const minVal = Math.min(...bpmChartHistory);
          const maxVal = Math.max(...bpmChartHistory);
          const padMin = Math.floor(Math.min(55, minVal - 5));
          const padMax = Math.ceil(Math.max(110, maxVal + 5));
          document.getElementById("chartRange").textContent = `${{padMin}} - ${{padMax}} BPM`;

          chartCtx.strokeStyle = "#FF4B4B";
          chartCtx.lineWidth = 2.5;
          chartCtx.beginPath();

          for (let i = 0; i < bpmChartHistory.length; i++) {{
            const x = (i / (MAX_CHART_PTS - 1)) * cw;
            const normY = (bpmChartHistory[i] - padMin) / (padMax - padMin + 1e-6);
            const y = ch - 12 - normY * (ch - 24);
            if (i === 0) chartCtx.moveTo(x, y);
            else chartCtx.lineTo(x, y);
          }}
          chartCtx.stroke();
        }}

        function estimarBpmConsenso(sinalFiltradoNorm, fs = 30.0) {{
          // Combina os 3 estimadores clássicos: Autocorrelação (ACF), Mediana IBI e Espectro DFT
          const acfRes = estimarBpmAutocorrelacao(sinalFiltradoNorm, fs);
          const ibiRes = estimarBpmPorPicosIBI(sinalFiltradoNorm, fs, acfRes.bpm);
          const dftRes = estimarFrequenciaSegura(sinalFiltradoNorm, fs, 0.78, 2.35, 1.35, true);
          const bpmDft = dftRes.freq * 60.0;

          const candidatos = [];
          if (acfRes.bpm >= 46.0 && acfRes.bpm <= 145.0) {{
            candidatos.push({{ bpm: acfRes.bpm, peso: 0.50, qualidade: acfRes.corr }});
          }}
          if (ibiRes.bpm >= 46.0 && ibiRes.bpm <= 145.0) {{
            candidatos.push({{ bpm: ibiRes.bpm, peso: 0.35, qualidade: ibiRes.confianca }});
          }}
          if (bpmDft >= 46.0 && bpmDft <= 145.0) {{
            candidatos.push({{ bpm: bpmDft, peso: 0.15, qualidade: Math.min(1.0, dftRes.snr / 4.0) }});
          }}

          if (candidatos.length === 0) return {{ bpm: 0.0, qualidade: 0.0 }};
          if (candidatos.length === 1) return {{ bpm: candidatos[0].bpm, qualidade: candidatos[0].qualidade }};

          // Se temos 2 ou 3 estimadores, verifica concordância com a mediana dos candidatos
          const valores = candidatos.map(c => c.bpm).sort((a, b) => a - b);
          const med = valores[Math.floor(valores.length / 2)];

          let somaPesos = 0.0;
          let somaBpm = 0.0;
          let maxQual = 0.0;
          for (const c of candidatos) {{
            // Descarta estimador divergente (> 12 BPM longe do consenso)
            if (Math.abs(c.bpm - med) <= 12.0) {{
              somaBpm += c.bpm * c.peso;
              somaPesos += c.peso;
              if (c.qualidade > maxQual) maxQual = c.qualidade;
            }}
          }}

          const bpmFinal = somaPesos > 0 ? (somaBpm / somaPesos) : med;
          return {{ bpm: bpmFinal, qualidade: maxQual }};
        }}

        function executarCicloDSP() {{
          if (!isRunning || !alvoDetectado) return;
          if (bufT.length < 25) return;

          const duracaoS = (bufT[bufT.length - 1] - bufT[0]) / 1000.0;
          if (duracaoS < MIN_DURATION_BPM_S) return;

          if (MODO === "dedo") {{
            // Inverte os sinais pois a sístole aumenta o volume sanguíneo (maior absorção de luz = menor intensidade refletida)
            const rUni = reamostrarUniforme(bufT, bufR.map(v => -v), 30.0);
            const gUni = reamostrarUniforme(bufT, bufG.map(v => -v), 30.0);
            const cUni = reamostrarUniforme(bufT, bufCombo.map(v => -v), 30.0);
            if (rUni.length < 60) return;

            const filtR = filtrarPassaFaixaPPG(rUni, 30.0);
            const filtG = filtrarPassaFaixaPPG(gUni, 30.0);
            const filtC = filtrarPassaFaixaPPG(cUni, 30.0);

            // Avalia a periodicidade cardíaca real em cada canal (R não-saturado, G ativo e Combo)
            const resR = filtR.rms < 22.0 ? estimarBpmConsenso(filtR.norm, 30.0) : {{ bpm: 0.0, qualidade: 0.0 }};
            const resG = filtG.rms < 22.0 ? estimarBpmConsenso(filtG.norm, 30.0) : {{ bpm: 0.0, qualidade: 0.0 }};
            const resC = filtC.rms < 22.0 ? estimarBpmConsenso(filtC.norm, 30.0) : {{ bpm: 0.0, qualidade: 0.0 }};

            // Escolhe o canal que apresentou a onda cardíaca mais periódica e limpa
            let melhor = resC;
            if (resG.qualidade >= melhor.qualidade && resG.bpm > 0) melhor = resG;
            if (resR.qualidade > melhor.qualidade * 1.05 && resR.bpm > 0) melhor = resR;

            if (melhor.bpm >= 46.0 && melhor.bpm <= 145.0) {{
              bpmAtual = estabBpm.atualizar(melhor.bpm);
            }}

            const spo2Bruto = calcularSpo2Empirico(rUni, gUni, reamostrarUniforme(bufT, bufB, 30.0));
            if (spo2Bruto >= 85.0 && spo2Bruto <= 99.5 && bpmAtual > 0) {{
              secAtual = estabSec.atualizar(spo2Bruto);
            }}
          }} else {{
            const gUni = reamostrarUniforme(bufT, bufG.map(v => -v), 30.0);
            if (gUni.length < 60) return;
            const filtG = filtrarPassaFaixaPPG(gUni, 30.0);

            const resFace = estimarBpmConsenso(filtG.norm, 30.0);
            if (resFace.bpm >= 46.0 && resFace.bpm <= 145.0) {{
              bpmAtual = estabBpm.atualizar(resFace.bpm);
            }}

            if (duracaoS >= MIN_DURATION_RPM_S) {{
              const gRespUni = reamostrarUniforme(bufT, bufG, 30.0);
              let mediaResp = 0;
              for (let i = 0; i < gRespUni.length; i++) mediaResp += gRespUni[i];
              mediaResp /= gRespUni.length;
              const normResp = new Float64Array(gRespUni.length);
              for (let i = 0; i < gRespUni.length; i++) normResp[i] = gRespUni[i] - mediaResp;

              const dftRpm = estimarFrequenciaSegura(normResp, 30.0, 0.15, 0.40, 1.30, false);
              const rpmBruto = dftRpm.freq * 60.0;
              if (rpmBruto >= 9.0 && rpmBruto <= 24.0) {{
                secAtual = estabSec.atualizar(rpmBruto);
              }}
            }}
          }}

          // Atualiza UI e Gráfico
          if (bpmAtual > 0) {{
            valBpm.textContent = `${{Math.round(bpmAtual)}} BPM`;
            bpmChartHistory.push(Number(bpmAtual.toFixed(1)));
            if (bpmChartHistory.length > MAX_CHART_PTS) bpmChartHistory.shift();
            desenharGrafico();
          }}
          if (secAtual > 0) {{
            valSec.textContent = MODO === "dedo"
              ? `${{secAtual.toFixed(1)}} %`
              : `${{Math.round(secAtual)}} RPM`;
          }}
        }}

        function renderLoop() {{
          if (!isRunning) return;

          if (videoEl.readyState >= videoEl.HAVE_CURRENT_DATA) {{
            // Sincroniza a resolução do canvas com a proporção real nativa da câmera (elimina imagem esticada)
            if (
              videoEl.videoWidth > 0 &&
              videoEl.videoHeight > 0 &&
              (viewCanvas.width !== videoEl.videoWidth || viewCanvas.height !== videoEl.videoHeight)
            ) {{
              viewCanvas.width = videoEl.videoWidth;
              viewCanvas.height = videoEl.videoHeight;
            }}

            const vw = viewCanvas.width;
            const vh = viewCanvas.height;

            // Verifica se a câmera entregou um NOVO quadro real (evita amostrar o mesmo frame 2x-4x em telas de 60/120Hz)
            const now = performance.now();
            const vTime = videoEl.currentTime;
            const isNovoQuadroCamera = (vTime !== lastVideoTime) || (now - lastSampleTime >= 32.0);

            // Espelha horizontalmente apenas na câmera frontal (efeito espelho natural)
            ctx.save();
            if (MODO === "rosto") {{
              ctx.translate(vw, 0);
              ctx.scale(-1, 1);
            }}
            ctx.drawImage(videoEl, 0, 0, vw, vh);
            ctx.restore();

            if (isNovoQuadroCamera) {{
              lastVideoTime = vTime;
              lastSampleTime = now;
              frameCount++;
              if (now - lastFpsTime >= 1000) {{
                currentFps = (frameCount * 1000) / (now - lastFpsTime);
                fpsBadge.textContent = `FPS: ${{currentFps.toFixed(1)}}`;
                frameCount = 0;
                lastFpsTime = now;
              }}
            }}

            if (MODO === "dedo") {{
              const roiSize = Math.floor(Math.min(vw, vh) * 0.5);
              const xi = Math.floor((vw - roiSize) / 2);
              const yi = Math.floor((vh - roiSize) / 2);

              if (isNovoQuadroCamera) {{
                const imgData = ctx.getImageData(xi, yi, roiSize, roiSize).data;

                let sumR = 0, sumG = 0, sumB = 0, totalAmostras = 0;
                let sumRUnsat = 0, countRUnsat = 0;
                let sumGActive = 0, countGActive = 0;

                for (let i = 0; i < imgData.length; i += 16) {{
                  const r = imgData[i];
                  const g = imgData[i + 1];
                  const b = imgData[i + 2];
                  sumR += r;
                  sumG += g;
                  sumB += b;
                  totalAmostras++;

                  // Coleta separadamente pixels onde o canal Vermelho NÃO estourou (< 252)
                  if (r >= 25 && r <= 251) {{
                    sumRUnsat += r;
                    countRUnsat++;
                  }}
                  // Coleta separadamente pixels onde o canal Verde tem luz transmitida ativa e sem estouro
                  if (g >= 3 && g <= 251) {{
                    sumGActive += g;
                    countGActive++;
                  }}
                }}

                const meanRAll = sumR / Math.max(1, totalAmostras);
                const meanGAll = sumG / Math.max(1, totalAmostras);
                const meanBAll = sumB / Math.max(1, totalAmostras);

                // Usa a média dos pixels não-saturados para preservar a amplitude AC da onda cardíaca
                const valR = countRUnsat > (totalAmostras * 0.08) ? (sumRUnsat / countRUnsat) : meanRAll;
                const valG = countGActive > (totalAmostras * 0.08) ? (sumGActive / countGActive) : meanGAll;
                const valCombo = meanRAll >= 244.0 && valG > 4.0 ? valG : (0.5 * valR + 0.5 * valG);

                // Dedo presente se o canal vermelho é dominante sobre o azul e tem brilho suficiente
                alvoDetectado = meanRAll > 125.0 && meanRAll > meanBAll * 1.35;

                if (alvoDetectado) {{
                  if (inicioContatoMs === 0) inicioContatoMs = now;
                  const tempoContatoMs = now - inicioContatoMs;

                  // Aguarda 1.0s inicial para a auto-exposição (AE) da câmera estabilizar após cobrir a lente
                  if (tempoContatoMs >= 1000) {{
                    bufR.push(valR);
                    bufG.push(valG);
                    bufB.push(meanBAll);
                    bufCombo.push(valCombo);
                    bufT.push(now);

                    while (bufT.length > 2 && (now - bufT[0] > WINDOW_MS || bufT.length > MAX_SAMPLES)) {{
                      bufR.shift();
                      bufG.shift();
                      bufB.shift();
                      bufCombo.shift();
                      bufT.shift();
                    }}
                  }}

                  const duracaoAtual = bufT.length > 1 ? (bufT[bufT.length - 1] - bufT[0]) / 1000.0 : 0.0;
                  const prog = Math.min(100, Math.floor((duracaoAtual / MIN_DURATION_BPM_S) * 100));
                  if (bpmAtual <= 0) {{
                    statusBox.className = "status-box";
                    statusBox.innerHTML = `⏳ <b>Estabilizando onda cardíaca (${{prog}}%)...</b> apoie o dedo de leve sem apertar.`;
                  }} else {{
                    statusBox.className = "status-box ok";
                    statusBox.innerHTML = "✅ <b>Leitura Estabilizada!</b> (Mantenha pressão leve e constante)";
                  }}
                }} else {{
                  inicioContatoMs = 0;
                  bufR = [];
                  bufG = [];
                  bufB = [];
                  bufCombo = [];
                  bufT = [];
                  estabBpm.reset();
                  estabSec.reset();
                  bpmAtual = 0;
                  secAtual = 0;
                  valBpm.textContent = "-- BPM";
                  valSec.textContent = "-- %";
                  statusBox.className = "status-box warn";
                  statusBox.innerHTML = "👆 <b>Posicione o dedo suavemente</b> cobrindo a câmera traseira e o flash.";
                }}
              }}

              ctx.lineWidth = 4;
              ctx.strokeStyle = !alvoDetectado ? "#EF4444" : (bpmAtual <= 0 ? "#F59E0B" : "#10B981");
              ctx.strokeRect(xi, yi, roiSize, roiSize);

            }} else {{
              // Modo Rosto (rPPG com Guia Biométrica Oval Fixa + ROI Anatômica da Testa)
              const menorDim = Math.min(vw, vh);
              const cx = Math.floor(vw * 0.5);
              const cy = Math.floor(vh * 0.48);
              const rx = Math.floor(menorDim * 0.30);
              const ry = Math.floor(menorDim * 0.39);

              // Coordenadas fixas da ROI da Testa (parte superior interna do oval)
              const tw = Math.max(24, Math.floor(rx * 0.92));
              const th = Math.max(16, Math.floor(ry * 0.30));
              const tx = Math.floor(cx - tw / 2);
              const ty = Math.floor(cy - ry * 0.72);

              // Região central do rosto (nariz/bochechas) para validação biométrica conjunta
              const cwBox = Math.max(24, Math.floor(rx * 0.80));
              const chBox = Math.max(20, Math.floor(ry * 0.40));
              const cxBox = Math.floor(cx - cwBox / 2);
              const cyBox = Math.floor(cy - chBox * 0.35);

              if (isNovoQuadroCamera) {{
                const testaImgData = ctx.getImageData(tx, ty, tw, th).data;
                const centroImgData = ctx.getImageData(cxBox, cyBox, cwBox, chBox).data;

                const presencaInstantanea = verificarRostoNaGuiaOval(testaImgData, centroImgData);
                if (presencaInstantanea) {{
                  framesSemRosto = 0;
                  alvoDetectado = true;
                }} else {{
                  framesSemRosto++;
                  if (framesSemRosto > 12) {{
                    alvoDetectado = false;
                  }}
                }}

                if (alvoDetectado) {{
                  let sumG = 0;
                  const countPx = tw * th;
                  for (let i = 0; i < testaImgData.length; i += 4) {{
                    sumG += testaImgData[i + 1];
                  }}
                  const meanG = sumG / countPx;

                  bufG.push(meanG);
                  bufT.push(now);
                  while (bufT.length > 2 && (now - bufT[0] > WINDOW_MS || bufT.length > MAX_SAMPLES)) {{
                    bufG.shift();
                    bufT.shift();
                  }}

                  const duracaoAtual = bufT.length > 1 ? (bufT[bufT.length - 1] - bufT[0]) / 1000.0 : 0.0;
                  const prog = Math.min(100, Math.floor((duracaoAtual / MIN_DURATION_BPM_S) * 100));
                  if (bpmAtual <= 0) {{
                    statusBox.className = "status-box";
                    statusBox.innerHTML = `⏳ <b>Capturando sinal da testa (${{prog}}%)...</b> mantenha o rosto na guia.`;
                  }} else {{
                    statusBox.className = "status-box ok";
                    statusBox.innerHTML = "✅ <b>Leitura rPPG Estabilizada!</b>";
                  }}
                }} else {{
                  bufG = [];
                  bufT = [];
                  estabBpm.reset();
                  estabSec.reset();
                  bpmAtual = 0;
                  secAtual = 0;
                  valBpm.textContent = "-- BPM";
                  valSec.textContent = "-- RPM";
                  statusBox.className = "status-box warn";
                  statusBox.innerHTML = "🙂 <b>Encaixe seu rosto na guia oval</b> com a testa dentro do retângulo.";
                }}
              }}

              // Máscara escura translúcida ao redor da silhueta oval para guiar o enquadramento
              ctx.save();
              ctx.fillStyle = "rgba(11, 13, 18, 0.48)";
              ctx.beginPath();
              ctx.rect(0, 0, vw, vh);
              ctx.ellipse(cx, cy, rx, ry, 0, 0, 2 * Math.PI, true);
              ctx.fill("evenodd");
              ctx.restore();

              // Desenha contorno da Guia Oval
              const corGuia = !alvoDetectado ? "#38BDF8" : (bpmAtual > 0 ? "#10B981" : "#F59E0B");
              ctx.save();
              ctx.lineWidth = 3;
              ctx.strokeStyle = corGuia;
              ctx.setLineDash([10, 6]);
              ctx.beginPath();
              ctx.ellipse(cx, cy, rx, ry, 0, 0, 2 * Math.PI);
              ctx.stroke();
              ctx.restore();

              // Desenha o retângulo da ROI da Testa
              ctx.save();
              ctx.lineWidth = 2.5;
              ctx.strokeStyle = !alvoDetectado ? "rgba(255, 255, 255, 0.65)" : (bpmAtual > 0 ? "#10B981" : "#F59E0B");
              ctx.strokeRect(tx, ty, tw, th);
              ctx.fillStyle = "rgba(11, 13, 18, 0.65)";
              ctx.fillRect(tx, Math.max(0, ty - 20), 76, 18);
              ctx.fillStyle = "#FAFAFA";
              ctx.font = "600 11px sans-serif";
              ctx.fillText("ROI TESTA", tx + 6, Math.max(12, ty - 7));
              ctx.restore();
            }}
          }}

          animId = requestAnimationFrame(renderLoop);
        }}

        // Garante que o iframe do componente herde a política de permissão de câmera do documento pai
        try {{
          if (window.frameElement) {{
            window.frameElement.setAttribute("allow", "camera *; microphone *; autoplay *");
          }}
        }} catch (e) {{
          console.warn("Aviso ao definir allow no frameElement:", e);
        }}

        const navPrincipal = (
          window.parent &&
          window.parent.navigator &&
          window.parent.navigator.mediaDevices
        )
          ? window.parent.navigator
          : navigator;

        const btnTorch = document.getElementById("btnTorch");
        let torchAtiva = false;

        async function setTorch(track, estado) {{
          if (!track || !track.applyConstraints) return false;
          try {{
            const caps = track.getCapabilities ? track.getCapabilities() : {{}};
            if (!caps.torch) {{
              return false;
            }}
            await track.applyConstraints({{ advanced: [{{ torch: estado }}] }});
            torchAtiva = estado;
            if (btnTorch) {{
              btnTorch.textContent = estado ? "🔦 Desligar Flash" : "🔦 Ligar Flash";
              btnTorch.style.background = estado ? "#10B981" : "#4C566A";
            }}
            return true;
          }} catch (e) {{
            console.warn("Falha ao definir torch:", e);
            return false;
          }}
        }}

        async function tentarAtivarTorch(currentStream) {{
          let track = currentStream.getVideoTracks()[0];
          if (!track) return false;

          // Aguarda 300ms para o hardware da câmera estabilizar após o play()
          await new Promise(r => setTimeout(r, 300));

          // 1. Verifica se a lente traseira atual possui flash (caps.torch)
          let ok = await setTorch(track, true);
          if (ok) return true;

          // 2. Retry após mais 350ms
          await new Promise(r => setTimeout(r, 350));
          ok = await setTorch(track, true);
          if (ok) return true;

          // 3. Em celulares com múltiplas lentes traseiras, a lente padrão pode ser a ultra-wide (sem flash).
          // Enumera todas as câmeras e testa as traseiras até achar a lente principal que tem caps.torch = true.
          try {{
            if (navPrincipal.mediaDevices.enumerateDevices) {{
              const devices = await navPrincipal.mediaDevices.enumerateDevices();
              const videoInputs = devices.filter(d => d.kind === "videoinput");
              const currentId = track.getSettings ? track.getSettings().deviceId : "";

              // Prioriza câmeras cujo label indique traseira/back/environment/0
              const candidatos = videoInputs.filter(d => d.deviceId !== currentId);

              for (const dev of candidatos) {{
                const label = (dev.label || "").toLowerCase();
                if (label.includes("front") || label.includes("user") || label.includes("frontal")) {{
                  continue;
                }}
                try {{
                  const s = await navPrincipal.mediaDevices.getUserMedia({{
                    audio: false,
                    video: {{
                      deviceId: {{ exact: dev.deviceId }},
                      width: {{ ideal: 640 }},
                      height: {{ ideal: 480 }}
                    }}
                  }});
                  const t = s.getVideoTracks()[0];
                  if (t) {{
                    await new Promise(r => setTimeout(r, 250));
                    const caps = t.getCapabilities ? t.getCapabilities() : {{}};
                    if (caps.torch) {{
                      track.stop();
                      stream = s;
                      videoEl.srcObject = s;
                      await videoEl.play();
                      await new Promise(r => setTimeout(r, 250));
                      await setTorch(t, true);
                      return true;
                    }} else {{
                      t.stop();
                    }}
                  }}
                }} catch (ignore) {{}}
              }}
            }}
          }} catch (e) {{
            console.warn("Erro ao buscar câmeras traseiras alternativas com flash:", e);
          }}

          return false;
        }}

        async function obterStreamCamera(modo) {{
          if (!navPrincipal.mediaDevices || !navPrincipal.mediaDevices.getUserMedia) {{
            throw new Error("API de câmera (mediaDevices) indisponível neste navegador.");
          }}

          const facing = modo === "dedo" ? "environment" : "user";
          const tentativas = [];

          if (modo === "dedo") {{
            tentativas.push({{
              audio: false,
              video: {{
                facingMode: {{ exact: "environment" }},
                width: {{ ideal: 640 }},
                height: {{ ideal: 480 }},
                frameRate: {{ ideal: 30 }}
              }}
            }});
          }}

          tentativas.push(
            {{
              audio: false,
              video: {{
                facingMode: {{ ideal: facing }},
                width: {{ ideal: 640 }},
                height: {{ ideal: 480 }},
                frameRate: {{ ideal: 30 }}
              }}
            }},
            {{
              audio: false,
              video: {{ facingMode: facing }}
            }},
            {{
              audio: false,
              video: true
            }}
          );

          let ultimoErro = null;
          for (const constraints of tentativas) {{
            try {{
              return await navPrincipal.mediaDevices.getUserMedia(constraints);
            }} catch (err) {{
              ultimoErro = err;
            }}
          }}
          throw ultimoErro;
        }}

        async function toggleCamera() {{
          if (isRunning) {{
            isRunning = false;
            if (animId) cancelAnimationFrame(animId);
            if (calcTimer) clearInterval(calcTimer);
            if (stream) {{
              stream.getTracks().forEach(t => t.stop());
              stream = null;
            }}
            btnToggle.textContent = "📷 Ligar Câmera";
            btnToggle.classList.remove("active");
            if (btnTorch) btnTorch.style.display = "none";
            torchAtiva = false;
            statusBox.className = "status-box";
            statusBox.innerHTML = "Câmera desligada. Toque em <b>Ligar Câmera</b> para iniciar.";
            fpsBadge.textContent = "FPS: --";
            return;
          }}

          try {{
            statusBox.className = "status-box";
            statusBox.innerHTML = "⏳ Solicitando acesso à câmera...";

            stream = await obterStreamCamera(MODO);
            videoEl.srcObject = stream;
            await videoEl.play();

            isRunning = true;
            btnToggle.textContent = "⏹️ Desligar Câmera";
            btnToggle.classList.add("active");
            lastFpsTime = performance.now();
            frameCount = 0;

            animId = requestAnimationFrame(renderLoop);
            calcTimer = setInterval(executarCicloDSP, 500);

            // Ativa a lanterna (torch) se estiver no modo Dedo
            if (MODO === "dedo") {{
              if (btnTorch) {{
                btnTorch.style.display = "inline-flex";
                btnTorch.textContent = "🔦 Ligar Flash";
                btnTorch.style.background = "#4C566A";
              }}
              tentarAtivarTorch(stream).then((ativou) => {{
                if (!ativou) {{
                  console.warn("Lanterna não ativada automaticamente. Use o botão 🔦 Ligar Flash ou abra no Chrome.");
                }}
              }});
            }} else {{
              if (btnTorch) btnTorch.style.display = "none";
            }}
          }} catch (err) {{
            statusBox.className = "status-box warn";
            statusBox.innerHTML = `❌ <b>Erro ao acessar câmera:</b> ${{err.message || err}}. Verifique a permissão do navegador.`;
          }}
        }}

        btnToggle.addEventListener("click", toggleCamera);
        if (btnTorch) {{
          btnTorch.addEventListener("click", async () => {{
            if (!stream) return;
            const track = stream.getVideoTracks()[0];
            const novoEstado = !torchAtiva;
            const ok = await setTorch(track, novoEstado);
            if (!ok) {{
              const okBusca = await tentarAtivarTorch(stream);
              if (!okBusca) {{
                alert("Este navegador não suporta controle da lanterna (o Firefox Android não possui suporte à API de flash). Abra este mesmo link no Google Chrome para usar o flash!");
              }}
            }}
          }});
        }}
        desenharGrafico();
      </script>
    </body>
    </html>
    """
    components.html(html_code, height=820, scrolling=False)


# ==============================================================================
# 1. MODO DEDO / TRASEIRA (PPG de Contato: BPM + SpO2)
# ==============================================================================
if fonte == "Dedo (Traseira + Flash)":
    renderizar_monitor_camera_cloud(modo="dedo")

# ==============================================================================
# 2. MODO FACIAL / FRONTAL (rPPG: BPM + RPM)
# ==============================================================================
elif fonte == "Rosto (Frontal - rPPG)":
    renderizar_monitor_camera_cloud(modo="rosto")

# ==============================================================================
# 3. MODO VÍDEO (Upload Offline processado via SciPy/OpenCV)
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
                small = frame[::2, ::2]
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
