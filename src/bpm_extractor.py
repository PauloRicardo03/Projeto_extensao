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

        if len(self.historico) >= 3:
            mediana_atual = float(np.median(self.historico))
            if abs(novo_valor - mediana_atual) > 0.25 * mediana_atual:
                novo_valor = mediana_atual + np.sign(novo_valor - mediana_atual) * (0.10 * mediana_atual)

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
    """Calcula SpO2 (%) pela técnica de Razão de Razões (Ros) entre os canais Vermelho e Azul."""
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
          max-height: 56vh;
          aspect-ratio: 4 / 3;
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
          width: 100%;
          height: 100%;
          object-fit: cover;
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
        <button id="btnToggle" class="btn-camera">📷 Ligar Câmera</button>
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

        // Buffers de sinal e timestamps
        let bufR = [];
        let bufG = [];
        let bufB = [];
        let bufT = [];
        const MAX_SAMPLES = MODO === "dedo" ? 180 : 300;
        const MIN_SAMPLES_BPM = MODO === "dedo" ? 75 : 100;
        const MIN_SAMPLES_RPM = 160;

        // Estado de FPS
        let frameCount = 0;
        let lastFpsTime = performance.now();
        let currentFps = 30.0;

        // Estado de detecção
        let alvoDetectado = false;
        let lastFaceBox = null;
        let framesSemRosto = 0;

        // Histórico para o gráfico contínuo
        let bpmChartHistory = [];
        const MAX_CHART_PTS = 120;

        // Classe Estabilizador Fisiológico (Mediana + Slew-Rate Limiter + EMA)
        class EstabilizadorFisiologico {{
          constructor(janelaMediana = 7, alpha = 0.15, maxDelta = 1.2) {{
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
            if (novoValor <= 0) return this.valorEstavel;
            if (this.historico.length >= 3) {{
              const med = this.mediana(this.historico);
              if (Math.abs(novoValor - med) > 0.25 * med) {{
                novoValor = med + Math.sign(novoValor - med) * (0.10 * med);
              }}
            }}
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

        const estabBpm = new EstabilizadorFisiologico(7, 0.15, 1.2);
        const estabSec = new EstabilizadorFisiologico(7, 0.15, 0.5);
        let bpmAtual = 0.0;
        let secAtual = 0.0;

        // Funções DSP (Interpolação Uniforme, Detrend, Filtro Passa-Faixa, FFT com Hann + Zero-Padding)
        function reamostrarUniforme(tempos, valores, fsAlvo = 30.0) {{
          if (tempos.length < 10) return valores;
          const duracao = (tempos[tempos.length - 1] - tempos[0]) / 1000.0;
          if (duracao <= 1.0) return valores;
          const nPts = Math.floor(duracao * fsAlvo);
          if (nPts < 16) return valores;

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

        function detrendNormalizar(arr) {{
          const n = arr.length;
          let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
          for (let i = 0; i < n; i++) {{
            sumX += i;
            sumY += arr[i];
            sumXY += i * arr[i];
            sumXX += i * i;
          }}
          const denom = n * sumXX - sumX * sumX;
          const slope = denom !== 0 ? (n * sumXY - sumX * sumY) / denom : 0;
          const intercept = (sumY - slope * sumX) / n;

          const out = new Float64Array(n);
          let sum2 = 0;
          for (let i = 0; i < n; i++) {{
            out[i] = arr[i] - (slope * i + intercept);
            sum2 += out[i] * out[i];
          }}
          const std = Math.sqrt(sum2 / n) + 1e-9;
          for (let i = 0; i < n; i++) out[i] /= std;
          return {{ norm: out, stdBruto: std }};
        }}

        function estimarFrequenciaSegura(sinalNorm, fs, fMin, fMax, snrMin = 1.55) {{
          const n = sinalNorm.length;
          if (n < 30) return 0.0;

          // Aplica janela de Hann e calcula espectro DFT de alta resolução (passo 0.01 Hz)
          const janelado = new Float64Array(n);
          for (let i = 0; i < n; i++) {{
            const w = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * i) / (n - 1)));
            janelado[i] = sinalNorm[i] * w;
          }}

          const passoHz = 0.01;
          const numBins = Math.floor((fMax - fMin) / passoHz) + 1;
          const mags = new Float64Array(numBins);
          let maxMag = -1.0;
          let bestFreq = 0.0;

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
              bestFreq = freq;
            }}
          }}

          // Calcula mediana espectral para validação de SNR
          const sortedMags = [...mags].sort((a, b) => a - b);
          const medMag = sortedMags[Math.floor(sortedMags.length / 2)] + 1e-9;
          const snr = maxMag / medMag;
          if (snr < snrMin) return 0.0;

          return bestFreq;
        }}

        function calcularSpo2Empirico(arrR, arrB) {{
          const n = arrR.length;
          let meanR = 0, meanB = 0;
          for (let i = 0; i < n; i++) {{
            meanR += arrR[i];
            meanB += arrB[i];
          }}
          meanR /= n;
          meanB /= n;
          if (meanR < 1e-3 || meanB < 1e-3) return 0.0;

          const detR = detrendNormalizar(arrR).stdBruto;
          const detB = detrendNormalizar(arrB).stdBruto;
          if (detB < 1e-4) return 0.0;

          let rOs = (detR / meanR) / (detB / meanB);
          rOs = Math.max(0.4, Math.min(1.2, rOs));
          const spo2 = 110.0 - 25.0 * rOs;
          return Math.max(70.0, Math.min(100.0, spo2));
        }}

        function detectarRostoYCrCb(imgData, w, h) {{
          const data = imgData.data;
          const mx = Math.floor(w * 0.14);
          const my = Math.floor(h * 0.08);
          let sumX = 0, sumY = 0, count = 0;
          let minX = w, maxX = 0, minY = h, maxY = 0;

          for (let y = my; y < h - my; y += 2) {{
            for (let x = mx; x < w - mx; x += 2) {{
              const idx = (y * w + x) * 4;
              const r = data[idx];
              const g = data[idx + 1];
              const b = data[idx + 2];

              const luma = 0.299 * r + 0.587 * g + 0.114 * b;
              const cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b;
              const cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b;

              if (luma > 40 && cr >= 133 && cr <= 178 && cb >= 77 && cb <= 128) {{
                count++;
                sumX += x;
                sumY += y;
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
              }}
            }}
          }}

          const areaTotalAmostrada = ((w - 2 * mx) * (h - 2 * my)) / 4;
          if (count < areaTotalAmostrada * 0.12) return null;

          const fw = maxX - minX;
          const fh = maxY - minY;
          if (fw < w * 0.18 || fh < h * 0.20) return null;

          return [minX, minY, fw, fh];
        }}

        function desenharGrafico() {{
          const cw = chartCanvas.width;
          const ch = chartCanvas.height;
          chartCtx.clearRect(0, 0, cw, ch);

          // Linhas de grade sutis
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

        function executarCicloDSP() {{
          if (!isRunning || !alvoDetectado) return;

          if (MODO === "dedo") {{
            if (bufR.length < MIN_SAMPLES_BPM) return;
            const rUni = reamostrarUniforme(bufT, bufR, 30.0);
            const {{ norm, stdBruto }} = detrendNormalizar(rUni);
            if (stdBruto > 28.0) return; // Descarta se houve movimento brusco do dedo

            const peakHz = estimarFrequenciaSegura(norm, 30.0, 0.85, 2.5, 1.55);
            const bpmBruto = peakHz * 60.0;
            const spo2Bruto = calcularSpo2Empirico(bufR, bufB);

            if (bpmBruto >= 50.0 && bpmBruto <= 150.0) {{
              bpmAtual = estabBpm.atualizar(bpmBruto);
            }}
            if (spo2Bruto >= 70.0 && spo2Bruto <= 100.0) {{
              secAtual = estabSec.atualizar(spo2Bruto);
            }}
          }} else {{
            if (bufG.length < MIN_SAMPLES_BPM) return;
            const gUni = reamostrarUniforme(bufT, bufG, 30.0);
            const {{ norm }} = detrendNormalizar(gUni);

            const peakBpmHz = estimarFrequenciaSegura(norm, 30.0, 0.85, 2.4, 1.45);
            const bpmBruto = peakBpmHz * 60.0;
            if (bpmBruto >= 50.0 && bpmBruto <= 145.0) {{
              bpmAtual = estabBpm.atualizar(bpmBruto);
            }}

            if (bufG.length >= MIN_SAMPLES_RPM) {{
              const peakRpmHz = estimarFrequenciaSegura(norm, 30.0, 0.15, 0.40, 1.30);
              const rpmBruto = peakRpmHz * 60.0;
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
            const vw = viewCanvas.width;
            const vh = viewCanvas.height;
            ctx.drawImage(videoEl, 0, 0, vw, vh);

            const now = performance.now();
            frameCount++;
            if (now - lastFpsTime >= 1000) {{
              currentFps = (frameCount * 1000) / (now - lastFpsTime);
              fpsBadge.textContent = `FPS: ${{currentFps.toFixed(1)}}`;
              frameCount = 0;
              lastFpsTime = now;
            }}

            if (MODO === "dedo") {{
              const roiSize = Math.floor(Math.min(vw, vh) * 0.5);
              const xi = Math.floor((vw - roiSize) / 2);
              const yi = Math.floor((vh - roiSize) / 2);
              const imgData = ctx.getImageData(xi, yi, roiSize, roiSize).data;

              let sumR = 0, sumB = 0;
              const totalPx = roiSize * roiSize;
              for (let i = 0; i < imgData.length; i += 16) {{
                sumR += imgData[i];
                sumB += imgData[i + 2];
              }}
              const amostras = totalPx / 4;
              const meanR = sumR / amostras;
              const meanB = sumB / amostras;

              alvoDetectado = meanR > 165.0;

              if (alvoDetectado) {{
                bufR.push(meanR);
                bufB.push(meanB);
                bufT.push(now);
                if (bufR.length > MAX_SAMPLES) {{
                  bufR.shift();
                  bufB.shift();
                  bufT.shift();
                }}
                const prog = Math.min(100, Math.floor((bufR.length / MIN_SAMPLES_BPM) * 100));
                if (bpmAtual <= 0) {{
                  statusBox.className = "status-box";
                  statusBox.innerHTML = `⏳ <b>Estabilizando leitura (${{prog}}%)...</b> mantenha o dedo parado.`;
                }} else {{
                  statusBox.className = "status-box ok";
                  statusBox.innerHTML = "✅ <b>Leitura Estabilizada!</b>";
                }}
              }} else {{
                bufR = [];
                bufB = [];
                bufT = [];
                estabBpm.reset();
                estabSec.reset();
                bpmAtual = 0;
                secAtual = 0;
                valBpm.textContent = "-- BPM";
                valSec.textContent = "-- %";
                statusBox.className = "status-box warn";
                statusBox.innerHTML = "👆 <b>Posicione o dedo</b> cobrindo a câmera traseira e o flash.";
              }}

              ctx.lineWidth = 4;
              ctx.strokeStyle = !alvoDetectado ? "#EF4444" : (bpmAtual <= 0 ? "#F59E0B" : "#10B981");
              ctx.strokeRect(xi, yi, roiSize, roiSize);

            }} else {{
              // Modo Rosto (rPPG)
              const fullData = ctx.getImageData(0, 0, vw, vh);
              const box = detectarRostoYCrCb(fullData, vw, vh);

              if (box) {{
                if (!lastFaceBox) {{
                  lastFaceBox = box;
                }} else {{
                  lastFaceBox = [
                    0.25 * box[0] + 0.75 * lastFaceBox[0],
                    0.25 * box[1] + 0.75 * lastFaceBox[1],
                    0.25 * box[2] + 0.75 * lastFaceBox[2],
                    0.25 * box[3] + 0.75 * lastFaceBox[3],
                  ];
                }}
                framesSemRosto = 0;
              }} else {{
                framesSemRosto++;
                if (framesSemRosto > 12) lastFaceBox = null;
              }}

              if (lastFaceBox) {{
                alvoDetectado = true;
                const [fx, fy, fw, fh] = lastFaceBox.map(Math.floor);
                const tx = Math.max(0, Math.floor(fx + 0.3 * fw));
                const ty = Math.max(0, Math.floor(fy + 0.1 * fh));
                const tw = Math.max(10, Math.floor(0.4 * fw));
                const th = Math.max(10, Math.floor(0.2 * fh));

                const testaData = ctx.getImageData(tx, ty, tw, th).data;
                let sumG = 0;
                const countPx = (tw * th);
                for (let i = 0; i < testaData.length; i += 4) {{
                  sumG += testaData[i + 1];
                }}
                const meanG = sumG / countPx;

                bufG.push(meanG);
                bufT.push(now);
                if (bufG.length > MAX_SAMPLES) {{
                  bufG.shift();
                  bufT.shift();
                }}

                const prog = Math.min(100, Math.floor((bufG.length / MIN_SAMPLES_BPM) * 100));
                if (bpmAtual <= 0) {{
                  statusBox.className = "status-box";
                  statusBox.innerHTML = `⏳ <b>Estabilizando rPPG da testa (${{prog}}%)...</b> evite movimentos.`;
                }} else {{
                  statusBox.className = "status-box ok";
                  statusBox.innerHTML = "✅ <b>Leitura rPPG Estabilizada!</b>";
                }}

                // Retângulo do rosto (Azul) e ROI da testa (Amarelo/Verde)
                ctx.lineWidth = 2.5;
                ctx.strokeStyle = "#38BDF8";
                ctx.strokeRect(fx, fy, fw, fh);

                ctx.lineWidth = 3;
                ctx.strokeStyle = bpmAtual > 0 ? "#10B981" : "#F59E0B";
                ctx.strokeRect(tx, ty, tw, th);
              }} else {{
                alvoDetectado = false;
                bufG = [];
                bufT = [];
                estabBpm.reset();
                estabSec.reset();
                bpmAtual = 0;
                secAtual = 0;
                valBpm.textContent = "-- BPM";
                valSec.textContent = "-- RPM";
                statusBox.className = "status-box warn";
                statusBox.innerHTML = "🙂 <b>Posicione seu rosto</b> centralizado em local bem iluminado.";
              }}
            }}
          }}

          animId = requestAnimationFrame(renderLoop);
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
            statusBox.className = "status-box";
            statusBox.innerHTML = "Câmera desligada. Toque em <b>Ligar Câmera</b> para iniciar.";
            fpsBadge.textContent = "FPS: --";
            return;
          }}

          try {{
            statusBox.className = "status-box";
            statusBox.innerHTML = "⏳ Abrindo câmera...";

            const constraints = {{
              audio: false,
              video: {{
                facingMode: {{ ideal: MODO === "dedo" ? "environment" : "user" }},
                width: {{ ideal: 640 }},
                height: {{ ideal: 480 }},
                frameRate: {{ ideal: 30 }}
              }}
            }};

            stream = await navigator.mediaDevices.getUserMedia(constraints);
            videoEl.srcObject = stream;
            await videoEl.play();

            // Ativa a lanterna (torch) se estiver no modo Dedo
            if (MODO === "dedo") {{
              const track = stream.getVideoTracks()[0];
              if (track && track.applyConstraints) {{
                try {{
                  await track.applyConstraints({{ advanced: [{{ torch: true }}] }});
                }} catch (torchErr) {{
                  console.warn("Lanterna não suportada neste dispositivo/navegador:", torchErr);
                }}
              }}
            }}

            isRunning = true;
            btnToggle.textContent = "⏹️ Desligar Câmera";
            btnToggle.classList.add("active");
            lastFpsTime = performance.now();
            frameCount = 0;

            animId = requestAnimationFrame(renderLoop);
            calcTimer = setInterval(executarCicloDSP, 500);
          }} catch (err) {{
            statusBox.className = "status-box warn";
            statusBox.innerHTML = `❌ <b>Erro ao acessar câmera:</b> ${{err.message || err}}. Verifique a permissão do navegador.`;
          }}
        }}

        btnToggle.addEventListener("click", toggleCamera);
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
