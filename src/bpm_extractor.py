# ==============================
# Importação das bibliotecas
# ==============================
import streamlit as st # Biblioteca para criar o aplicativo web e a interface.
import cv2 # OpenCV, para processamento de imagem e vídeo (ex: desenhar retângulo).
import numpy as np # NumPy, para cálculos numéricos e manipulação de arrays.
import threading # Para executar o cálculo de BPM em paralelo (thread) sem travar a interface.
import time # Para calcular o tempo (FPS) e adicionar pausas (sleep).
# Importa componentes para ligar a câmera (WebRTC) do navegador ao Python:
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase 
# webrtc_streamer: Cria o componente de vídeo na tela.
# VideoProcessorBase: Classe base que usamos para processar os frames de vídeo.
import av # Biblioteca que "traduz" os frames de vídeo entre o WebRTC e o NumPy/CV2.
# Importa funções de processamento de sinal do SciPy:
from scipy.signal import butter, filtfilt, find_peaks
# butter: Cria os coeficientes de um filtro digital (passa-banda).
# filtfilt: Aplica o filtro ao sinal (remove ruído) sem atraso de fase.
# find_peaks: Usado no modo vídeo para achar os picos (batidas).
# Importa funções de Transformada de Fourier (FFT) do SciPy:
from scipy.fft import rfft, rfftfreq
# rfft: Calcula a FFT de um sinal real (transforma o sinal do domínio do TEMPO para o domínio da FREQUÊNCIA).
# rfftfreq: Gera o "eixo X" (as frequências em Hz) correspondentes ao resultado do rfft.
import matplotlib.pyplot as plt # (Não está sendo usado neste código, mas serve para plotar gráficos).
import tempfile # Usado para criar um arquivo temporário para salvar o vídeo do upload.
import os # Usado para apagar o arquivo de vídeo temporário após a análise.


# ==============================
# Configuração inicial do Streamlit
# ==============================
# Configura a página do Streamlit (título da aba, layout).
st.set_page_config(page_title="PPG Real-Time + Flash", layout="centered")
st.title("💡 Monitor Cardíaco em Tempo Real") # Título principal na página.

# Injeta CSS personalizado para estilizar a página.
st.markdown(
    """
    <style>
    /* Esconde o texto de status "RUNNING" do componente WebRTC */
    div[data-testid="stWebRTCStatus"] { 
        display: none !important;
    }
    /* Estilização do player de vídeo */
    video {
        transform: rotate(90deg); /* Gira o vídeo (para celular em pé) */
        object-fit: cover; /* Garante que o vídeo preencha o espaço */
        width: 100%;
        height: 80vh; /* Altura de 80% da tela */
        border-radius: 10px; /* Bordas arredondadas */
    }
    </style>
    """,
    unsafe_allow_html=True, # Permite que o Streamlit injete o CSS.
)


# ==============================
# Seleção da fonte (Câmera ou Vídeo)
# ==============================
# Cria botões de rádio para o usuário escolher o modo de operação.
fonte = st.radio("Fonte do sinal", ("Câmera (WebRTC)",
                                  "Vídeo (upload)"), horizontal=True)


# ==============================
# Série de BPM (histórico)
# ==============================
# O st.session_state é a "memória" do app que persiste entre os recarregamentos.
# Verifica se a lista 'bpm_series' (histórico do gráfico) já existe na memória.
if "bpm_series" not in st.session_state:
    st.session_state.bpm_series = [] # Se não existir, cria como uma lista vazia.
MAX_POINTS = 180  # Define o tamanho máximo do histórico do gráfico (aprox. 3 min se 1 ponto/seg).


# ==============================
# Funções utilitárias
# ==============================
# Define a função do filtro passa-banda (deixa passar só frequências entre lowcut e highcut).
def bandpass_filter(data, lowcut, highcut, fs):
    # data: O sinal bruto (array numpy).
    # lowcut: Frequência de corte inferior (ex: 0.7 Hz).
    # highcut: Frequência de corte superior (ex: 3.0 Hz).
    # fs: Taxa de amostragem (Frames Por Segundo).

    # Trava de segurança (Teorema de Nyquist): fs deve ser pelo menos 2x o highcut.
    if fs <= 2 * highcut:
        fs = 2 * highcut + 1.0  # Força um fs seguro se o FPS estiver muito baixo.

    nyq = 0.5 * fs # Calcula a frequência de Nyquist (metade do fs).
    low = lowcut / nyq # Normaliza a frequência de corte inferior (valor entre 0 e 1).
    high = highcut / nyq # Normaliza a frequência de corte superior.

    # Garante que os cortes estão estritamente entre 0 e 1 (requisito do 'butter').
    low = max(0.01, min(0.99, low))
    high = max(0.01, min(0.99, high))

    # Desenha o filtro digital, obtendo os coeficientes (b, a).
    b, a = butter(2, [low, high], btype="band")
    # Aplica o filtro (b, a) aos dados (data) e retorna o sinal limpo.
    return filtfilt(b, a, data)


# ==============================
# MODO CÂMERA (WebRTC)
# ==============================
# Verifica qual botão de rádio o usuário selecionou.
if fonte == "Câmera (WebRTC)":

    # --- Controle do Botão Toggle ---
    TOGGLE_KEY = "camera_on_v5" # Chave única para o botão na memória (session_state).
    # Verifica se o estado do botão (ligado/desligado) já existe na memória.
    if TOGGLE_KEY not in st.session_state:
        st.session_state[TOGGLE_KEY] = False # Se não, define como desligado por padrão.
    # Cria o botão toggle, 'camera_on' guardará True (ligado) ou False (desligado).
    camera_on = st.toggle("📷 Ligar/Desligar Câmera", key=TOGGLE_KEY)

    st.divider() # Desenha uma linha horizontal na interface.

    # --- Área de Informações (Placeholders) ---
    # Cria colunas na interface (2/3 para status, 1/3 para BPM).
    col_status, col_bpm = st.columns([2, 1])

    with col_status:
        # 'st.empty()' cria um placeholder (espaço vazio) para atualizar o status.
        status_display = st.empty()

    with col_bpm:
        # Placeholder para atualizar o valor do BPM.
        bpm_display = st.empty()

    # ----------------------------------------------

    # --- Processador de Vídeo ---
    # Define nossa classe de processamento, herdando de VideoProcessorBase.
    class MeuProcessadorDeVideo(VideoProcessorBase):
        # O 'init' (construtor) é chamado uma vez quando o processador é criado.
        def __init__(self):
            # 'buffer_R': Lista que armazena os valores médios do canal vermelho (sinal bruto).
            self.buffer_R = []
            # 'buffer_size': Quantidade de amostras (frames) que o buffer deve guardar (150).
            self.buffer_size = 150
            # 'lock': Um "cadeado" para evitar que a thread de cálculo e a thread de vídeo
            # mexam no 'buffer_R' ao mesmo tempo (evita "race condition").
            self.lock = threading.Lock()
            # 'bpm': Armazena o valor de BPM calculado (inicialmente 0).
            self.bpm = 0.0
            # 'fps': Armazena os frames por segundo calculados.
            self.fps = 0.0
            # 'frame_count': Contador de frames para o cálculo de FPS.
            self.frame_count = 0
            # 'start_time': Tempo inicial para o cálculo de FPS.
            self.start_time = time.time()
            # 'dedo_detectado': Flag (True/False) que indica se o dedo está na câmera.
            self.dedo_detectado = False
            # 'RED_THRESHOLD': O nível mínimo de "vermelho" para considerar o dedo detectado.
            self.RED_THRESHOLD = 220
            # 'calc_thread': Define a thread de cálculo.
            # 'target': Função que a thread deve executar (o loop 'calcula_bpm_continuo').
            # 'daemon=True': Faz a thread fechar automaticamente quando o programa principal parar.
            self.calc_thread = threading.Thread(
                target=self.calcula_bpm_continuo, daemon=True)
            # Inicia a thread de cálculo em segundo plano.
            self.calc_thread.start()

        # Esta função roda em loop infinito na thread de cálculo (em paralelo).
        def calcula_bpm_continuo(self):
            while True: # Loop infinito.
                time.sleep(4.0)  # Pausa por 4 segundos antes de recalcular o BPM.

                # 'with self.lock:' Adquire o "cadeado". Garante que 'recv' não mude o buffer.
                with self.lock:
                    # Se o dedo não estiver detectado ou o buffer não estiver cheio (150 amostras),
                    if not self.dedo_detectado or len(self.buffer_R) < self.buffer_size:
                        continue # ...pula este ciclo e volta a dormir por 4s.

                    # --- Trava de Segurança (Nyquist) ---
                    fs_real = self.fps # Pega o FPS real medido.
                    # Se o FPS real for muito baixo (< 7), usa 30.0 (padrão) no cálculo.
                    # Isso evita o crash 'ValueError' do filtro 'butter'.
                    fs = fs_real if fs_real > 7.0 else 30.0

                    # Bloco 'try' para capturar erros matemáticos (ex: 'ValueError') sem travar.
                    try:
                        # Converte a lista 'buffer_R' para um array numpy.
                        data = np.array(self.buffer_R)
                        # Normalização (Z-score): (valor - média) / desvio_padrão.
                        # Isso centraliza o sinal em 0 e remove variações lentas de brilho.
                        data = (data - np.mean(data)) / (np.std(data) + 1e-9)

                        # Aplica o filtro passa-banda (deixa passar só 0.7Hz a 3.0Hz).
                        filtered = bandpass_filter(data, 0.7, 3.0, fs)

                        # --- Cálculo da FFT ---
                        # Gera o "eixo X" (frequências em Hz) para o resultado da FFT.
                        freqs = rfftfreq(len(filtered), d=1.0 / fs)
                        # Calcula a FFT (transforma o sinal em frequências) e pega a magnitude.
                        fft_vals = np.abs(rfft(filtered))

                        # --- Encontrar o Pico ---
                        # Cria uma máscara booleana, selecionando só a faixa de interesse (0.7-3.0Hz).
                        valid = (freqs >= 0.7) & (freqs <= 3.0)
                        # Verifica se existe algum valor na faixa de interesse.
                        if np.any(valid):
                            # Encontra a frequência (Hz) que tem a maior magnitude (pico)
                            # *apenas* dentro da faixa 'valid'.
                            peak_freq = freqs[valid][np.argmax(
                                fft_vals[valid])]
                            # Converte a frequência de pico (Hz = batidas/seg) para BPM (batidas/min).
                            bpm = peak_freq * 60.0
                            # Validação: só aceita BPMs entre 40 e 180.
                            if 40 < bpm < 180:
                                # Suavização Exponencial (EMA):
                                # O novo BPM é 20% do valor recém-calculado + 80% do valor antigo.
                                # Isso estabiliza o número e evita que ele "pule" muito.
                                self.bpm = bpm if self.bpm == 0 else bpm*0.2 + self.bpm*0.8
                    except Exception as e: # Se der qualquer erro matemático...
                        # Imprime o erro no console (terminal) mas não para o programa.
                        print(f"Erro no cálculo BPM: {e}")
                        continue # Pula este ciclo de cálculo.

        # Esta função ('recv') é chamada para CADA frame que chega da câmera.
        def recv(self, frame):
            # Converte o frame do formato 'av' (WebRTC) para 'numpy' (OpenCV).
            img = frame.to_ndarray(format="bgr24")
            # Pega as dimensões da imagem (Altura, Largura, Canais).
            h, w, _ = img.shape
            # Define o tamanho da Região de Interesse (ROI) como 50% da menor dimensão.
            roi_size = int(min(h, w) * 0.5)
            # Calcula a coordenada X inicial para centralizar o ROI.
            x_i = (w - roi_size) // 2
            # Calcula a coordenada Y inicial para centralizar o ROI.
            y_i = (h - roi_size) // 2
            # Recorta o quadrado (ROI) do centro da imagem original.
            roi = img[y_i:y_i + roi_size, x_i:x_i + roi_size]

            # --- Cálculo de FPS (Frames Por Segundo) ---
            self.frame_count += 1 # Incrementa o contador de frames.
            now = time.time() # Pega o tempo atual.
            # Verifica se já se passou 1 segundo.
            if now - self.start_time >= 1.0:
                # Calcula o FPS (frames / segundos).
                self.fps = self.frame_count / (now - self.start_time)
                self.frame_count = 0 # Zera o contador de frames.
                self.start_time = now # Reseta o tempo inicial.

            # --- Processamento do Sinal (Sinal Bruto) ---
            # Calcula a média de todos os pixels do canal VERMELHO (índice 2 no BGR) no ROI.
            mean_red_value = np.mean(roi[:, :, 2]) if roi.size > 0 else 0
            # Verifica se a média de vermelho é alta o suficiente para ser um dedo.
            self.dedo_detectado = mean_red_value > self.RED_THRESHOLD

            # 'with self.lock:' Adquire o "cadeado". Garante que 'calcula_bpm' não leia o buffer.
            with self.lock:
                if self.dedo_detectado: # Se o dedo está na câmera...
                    # Adiciona a média de vermelho atual ao final do buffer.
                    self.buffer_R.append(mean_red_value)
                    # Se o buffer ultrapassar o tamanho máximo (150)...
                    if len(self.buffer_R) > self.buffer_size:
                        self.buffer_R.pop(0) # Remove o valor mais antigo (do começo da lista).
                else: # Se o dedo foi removido...
                    self.buffer_R.clear() # Limpa o buffer (evita dados sujos).
                    self.bpm = 0.0 # Zera o BPM.

            # --- Feedback Visual (Desenho no Frame) ---
            if not self.dedo_detectado: # Se não há dedo...
                cor_retangulo = (0, 0, 255)  # Vermelho (BGR).
            elif self.bpm <= 0: # Se há dedo, mas ainda calculando...
                cor_retangulo = (0, 255, 255)  # Amarelo (BGR).
            else: # Se há dedo e BPM foi calculado...
                cor_retangulo = (0, 255, 0)  # Verde (BGR).

            # Desenha o retângulo (na imagem 'img') com a cor definida.
            cv2.rectangle(img, (x_i, y_i), (x_i + roi_size,
                                          y_i + roi_size), cor_retangulo, 3)
            # Converte o frame modificado ('img') de volta para o formato 'av' e o retorna.
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    # --- Configurações da Câmera (WebRTC) ---
    video_constraints = {
        "video": {
            # Pede para o navegador usar a câmera traseira ("environment").
            "facingMode": {"ideal": "environment"},
            "width": {"ideal": 640}, # Sugere largura.
            "height": {"ideal": 480}, # Sugere altura.
            "frameRate": {"ideal": 30}, # Sugere 30 FPS.
            "torch": True, # Pede para ligar o flash (lanterna).
        },
        "audio": False, # Não precisamos de áudio.
    }

    # --- Inicialização do Componente WebRTC ---
    # 'ctx' (contexto) armazena o estado do componente de vídeo.
    ctx = webrtc_streamer(
        key="camera_flash", # ID único do componente.
        # Fábrica: Diz ao webrtc para criar um objeto da nossa classe 'MeuProcessadorDeVideo'.
        video_processor_factory=MeuProcessadorDeVideo,
        # Aplica as restrições de vídeo (câmera traseira, flash).
        media_stream_constraints=video_constraints,
        # 'True' faz o 'recv' rodar em sua própria thread, não travando a UI.
        async_processing=True,
        # Conecta o estado do componente (ligado/desligado) ao nosso botão toggle.
        desired_playing_state=camera_on,
    )

    # ===============================================
    # ATUALIZAÇÃO DA INTERFACE (Roda a cada 1 segundo)
    # ===============================================
    # Verifica se o componente de vídeo (ctx) está rodando e se o processador já foi criado.
    if ctx and ctx.state.playing and ctx.video_processor:
        # Pega uma referência ao nosso processador (o objeto 'MeuProcessadorDeVideo').
        vp = ctx.video_processor

        # Pega o valor do BPM de dentro do processador de forma segura.
        bpm_val = float(vp.bpm) if vp.bpm and vp.bpm > 0 else 0.0
        # Pega o status do dedo de forma segura.
        dedo_ok = getattr(vp, 'dedo_detectado', False)

        # --- Atualiza os Placeholders ---
        if not dedo_ok: # Se o dedo não está na câmera...
            # Atualiza o placeholder 'status_display' com um aviso.
            status_display.warning(
                "👆 **Posicione o dedo** cobrindo a câmera e o flash.")
        elif bpm_val <= 0: # Se o dedo está, mas o BPM é zero (calculando)...
            status_display.info("⏳ **Calculando...** mantenha o dedo parado.")
        else: # Se o BPM é válido...
            status_display.success("✅ **Leitura Estável!**")

        if bpm_val > 0: # Se o BPM é válido...
            # Atualiza o placeholder 'bpm_display' com o valor e um coração.
            bpm_display.metric(label="BPM", value=f"{bpm_val:.0f}", delta="❤️")
            # Adiciona o valor ao histórico do gráfico.
            st.session_state.bpm_series.append(bpm_val)
            # Se o histórico (gráfico) ficou muito longo...
            if len(st.session_state.bpm_series) > MAX_POINTS:
                # Remove os pontos mais antigos (mantém apenas os últimos 180).
                st.session_state.bpm_series = st.session_state.bpm_series[-MAX_POINTS:]
        else: # Se o BPM é zero...
            # Mostra "--" no placeholder 'bpm_display'.
            bpm_display.metric(label="BPM", value="--", delta_color="off")

    else: # Se a câmera não está ligada...
        status_display.info("Aguardando inicialização da câmera...")
        bpm_display.metric(label="BPM", value="--")

    # --- Desenho do Gráfico ---
    st.subheader("Histórico") # Título da seção do gráfico.
    bpm_chart_placeholder = st.empty() # Placeholder para o gráfico.

    # Se houver pelo menos 1 ponto no histórico...
    if len(st.session_state.bpm_series) >= 1:
        # Desenha o gráfico de linha no placeholder.
        bpm_chart_placeholder.line_chart(st.session_state.bpm_series)
    else: # Se o histórico estiver vazio...
        bpm_chart_placeholder.caption("O gráfico aparecerá aqui...") # Mostra um texto.

    # --- Loop de Recarregamento (Rerun) ---
    time.sleep(4.0) # Pausa a thread principal do Streamlit por 4 segundos.
    st.rerun() # Força o script a rodar novamente do topo (atualizando a UI).

# ==============================
# MODO VÍDEO (upload)
# ==============================
else: # Se o usuário selecionou "Vídeo (upload)"...
    st.info("Envie um vídeo curto (ex.: .mp4) com o dedo cobrindo a câmera/lanterna.")
    # Cria o componente de upload de arquivo.
    file = st.file_uploader("Selecionar vídeo", type=[
                                  "mp4", "mov", "avi", "mkv"])
    # Checkbox para pular a detecção de dedo (útil se o vídeo só tiver o dedo).
    aplicar_limiar = st.checkbox(
        "Aplicar limiar de dedo (vermelho alto)", value=False)
    # Botão para iniciar a análise.
    iniciar = st.button("▶️ Analisar vídeo")

    # Placeholders para os resultados.
    bpm_video_placeholder = st.empty()
    debug_placeholder = st.empty()
    frame_preview = st.empty()

    # Se o usuário clicou em "Analisar" E um arquivo foi enviado...
    if iniciar and file is not None:
        # Cria um arquivo temporário de forma segura (para o OpenCV poder lê-lo).
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
            tmp.write(file.read()) # Escreve os dados do upload no arquivo temporário.
            temp_path = tmp.name # Salva o caminho do arquivo temporário.

        # Abre o arquivo de vídeo temporário com o OpenCV.
        cap = cv2.VideoCapture(temp_path)
        # Pega o FPS do arquivo de vídeo (ou usa 30 como padrão).
        fps_file = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # Demais variáveis de estado para a análise.
        roi_buffer_R = []
        bpm_display_video = 0.0
        alpha = 0.2 # Fator de suavização (EMA).
        RED_THRESHOLD = 10 # Limiar de vermelho (mais baixo para vídeos).
        # Limita a análise aos primeiros 1200 frames (aprox. 40s @ 30fps).
        max_frames = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1200, 1200))
        bpm_series_video = [] # Histórico do gráfico para o vídeo.

        # Loop 'for' que processa o vídeo frame a frame (até 'max_frames').
        for i in range(max_frames):
            ok, frame = cap.read() # Lê um frame.
            if not ok: # Se 'ok' for False, o vídeo acabou.
                break # Para o loop.

            # Cria uma versão pequena do frame para mostrar na tela (preview).
            small = cv2.resize(frame, (320, 240))
            # Mostra o preview no placeholder (convertendo de BGR para RGB).
            frame_preview.image(cv2.cvtColor(
                small, cv2.COLOR_BGR2RGB), caption=f"Frame {i+1}")

            # Lógica de extração do ROI (igual ao modo câmera).
            h, w, _ = frame.shape
            roi_size = int(min(h, w) * 0.5)
            x_i = (w - roi_size) // 2
            y_i = (h - roi_size) // 2
            roi = frame[y_i:y_i + roi_size, x_i:x_i + roi_size]

            # Extrai a média de vermelho (sinal bruto).
            mean_red = float(np.mean(roi[:, :, 2])) if roi.size > 0 else 0.0
            # Verifica se o dedo está presente (ou se o checkbox foi desmarcado).
            dedo_ok = (mean_red > RED_THRESHOLD) if aplicar_limiar else True

            if dedo_ok: # Se o sinal é válido...
                roi_buffer_R.append(mean_red) # Adiciona ao buffer.
                if len(roi_buffer_R) > 150: # Mantém o buffer com 150 amostras.
                    roi_buffer_R.pop(0)
                if len(roi_buffer_R) >= 60: # Se já tivermos dados suficientes...
                    data = np.array(roi_buffer_R, dtype=np.float64)
                    data = (data - np.mean(data)) / (np.std(data) + 1e-9) # Normaliza.
                    
                    try:
                        # Trava de segurança do FPS (igual ao modo câmera).
                        safe_fps = fps_file if fps_file > 6.0 else 30.0
                        # Aplica o filtro passa-banda.
                        filtered = bandpass_filter(data, 0.7, 3.0, safe_fps)
                        
                        # --- MÉTODO DE CÁLCULO (Domínio do Tempo) ---
                        # (Diferente do modo câmera, que usa FFT)
                        # Encontra os picos (batidas) no sinal já filtrado.
                        peaks, _ = find_peaks(
                            filtered, distance=int(0.3*safe_fps)) # Distância mínima de 0.3s entre picos.
                        
                        if len(peaks) >= 2: # Se encontrou pelo menos 2 picos...
                            # Calcula a diferença (em amostras) entre os picos.
                            # 'np.diff' calcula a diferença entre picos consecutivos.
                            # Divide pelo FPS para ter o tempo (em segundos) entre batidas (Intervalo R-R).
                            rr = np.diff(peaks) / safe_fps
                            # Filtra intervalos R-R (batidas) muito rápidos ou lentos (0.3s a 2.0s).
                            rr = rr[(rr > 0.3) & (rr < 2.0)]
                            if len(rr) > 0: # Se sobrar algum intervalo válido...
                                # Calcula o BPM: 60 / (tempo médio entre batidas em segundos).
                                # Usa a 'mediana' que é mais robusta a picos falsos (outliers).
                                bpm_now = 60.0 / np.median(rr)
                                if 40 < bpm_now < 180: # Validação (40-180 BPM).
                                    # Aplica suavização exponencial (EMA).
                                    bpm_display_video = bpm_now if bpm_display_video == 0 else bpm_now * \
                                        alpha + bpm_display_video*(1-alpha)
                    except: # Se der erro matemático...
                        pass # Ignora e continua.

                    # A cada 10 frames, atualiza a interface.
                    if i % 10 == 0 and bpm_display_video > 0:
                        bpm_series_video.append(bpm_display_video) # Adiciona ao histórico.
                        # Redesenha o gráfico do vídeo.
                        bpm_video_placeholder.line_chart(bpm_series_video)
            
            time.sleep(0.01) # Pequena pausa para a UI não travar.
        
        cap.release() # Libera o arquivo de vídeo (fecha).
        try:
            os.remove(temp_path) # Tenta apagar o arquivo temporário do disco.
        except:
            pass # Se não conseguir, ignora.