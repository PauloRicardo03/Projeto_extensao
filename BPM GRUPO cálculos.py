import cv2  # Biblioteca para capturar vídeo da câmera
import numpy as np  # Biblioteca para cálculos numéricos
import time  # Para medir o tempo e calcular FPS
from scipy.signal import butter, filtfilt, find_peaks  # Para filtragem e detecção de picos

# ===================== CONFIGURAÇÕES =====================
Tamanho_tela = 100  # Quantidade de frames que vamos analisar por vez
url = "http://192.168.100.182:4747/video"  # IP da câmera no aplicativo
cap = cv2.VideoCapture(url)  # Captura o vídeo via IP

if not cap.isOpened():  # Verifica se a câmera foi encontrada
    print("Camera não Encontrada")
    exit()

contador_frames = 0
tempo_espera = time.time()
taxa_fps = 30  # Inicialmente assumimos 30 FPS

# Buffers para armazenar médias de cada canal de cor
buffer_R, buffer_G, buffer_B = [], [], []

# Variáveis para fixar BPM
bpm_fixado = None       # Para armazenar o BPM fixo
bpm_primeiros = []      # Para armazenar os primeiros 2-3 BPM detectados
medir_novamente = True  # Controla se deve medir novos batimentos

print("Coloque seu Dedo na camera do celular")
print("Pressione 'q' na janela de vídeo para sair.")

# ===================== FUNÇÃO DE FILTRAGEM =====================
def bandpass_filter(signal, low=0.8, high=3, fs=30):
    nyq = 0.5 * fs
    low /= nyq
    high /= nyq
    b, a = butter(2, [low, high], btype='band')
    return filtfilt(b, a, signal)

# ===================== LOOP PRINCIPAL =====================
while True:
    retorno, imagem = cap.read()  # Captura um frame da câmera

    if not retorno:
        print("A Stream do video foi perdida, tentando reconectar...")
        cap = cv2.VideoCapture(url)
        time.sleep(2)
        continue

    contador_frames += 1

    # ===================== CÁLCULO DE FPS =====================
    if contador_frames % 60 == 0:
        cont_final = time.time()
        tempo_decorrido = cont_final - tempo_espera
        if tempo_decorrido > 0:
            taxa_fps = contador_frames / tempo_decorrido
        contador_frames = 0
        tempo_espera = time.time()

    # ===================== DEFINIÇÃO DO ROI =====================
    (largura, altura) = imagem.shape[:2]
    tamanho_roi = int(largura * 0.5)

    x_inicio = (altura - tamanho_roi) // 2
    y_inicio = (largura - tamanho_roi) // 2
    x_fim = x_inicio + tamanho_roi
    y_fim = y_inicio + tamanho_roi

    x_inicio = max(0, x_inicio)
    y_inicio = max(0, y_inicio)
    x_fim = min(altura, x_fim)
    y_fim = min(largura, y_fim)

    # ===================== EXTRAÇÃO DE MÉDIA DE COR =====================
    if x_fim > x_inicio and y_fim > y_inicio:
        roi_central = imagem[y_inicio:y_fim, x_inicio:x_fim]
        cv2.rectangle(imagem, (x_inicio, y_inicio), (x_fim, y_fim), (0, 255, 0), 2)
        media_BGR = np.mean(roi_central, axis=(0, 1))
    else:
        cv2.rectangle(imagem, (0, 0), (largura // 4, altura // 4), (0, 0, 255), 2)
        media_BGR = np.mean(imagem, axis=(0, 1))

    # ===================== ATUALIZAÇÃO DE BUFFERS =====================
    buffer_B.append(media_BGR[0])
    buffer_G.append(media_BGR[1])
    buffer_R.append(media_BGR[2])

    if len(buffer_B) > Tamanho_tela:
        buffer_B.pop(0)
        buffer_G.pop(0)
        buffer_R.pop(0)

    # ===================== VERIFICAÇÃO SE DEDO ESTÁ NA CÂMERA =====================
    if len(buffer_B) == Tamanho_tela:
        media_B_recente = media_BGR[0]
        media_G_recente = media_BGR[1]
        media_R_recente = media_BGR[2]

        vermelho = 60  # nível mínimo de vermelho

        dedo_na_camera = (media_R_recente > media_G_recente + vermelho and
                          media_R_recente > media_B_recente + vermelho)

        if dedo_na_camera and medir_novamente:
            # ===================== PROCESSAMENTO DO SINAL =====================
            Gnorm = np.array(buffer_G) / (np.mean(buffer_G) + 1e-9)
            sinal_ac = Gnorm - np.mean(Gnorm)
            sinal_filtrado = bandpass_filter(sinal_ac, low=0.8, high=3, fs=taxa_fps)
            peaks, _ = find_peaks(sinal_filtrado, distance=taxa_fps*0.4)

            if len(peaks) > 1:
                intervalos = np.diff(peaks) / taxa_fps
                bpm_atual = 60 / np.mean(intervalos)

                # ===================== FIXAR PRIMEIROS BATIMENTOS =====================
                if bpm_fixado is None:
                    bpm_primeiros.append(bpm_atual)
                    if len(bpm_primeiros) >= 3:
                        bpm_fixado = np.mean(bpm_primeiros)

                valor_a_mostrar = bpm_fixado if bpm_fixado is not None else bpm_atual
                info_bpm = f"O seu BPM está em: {valor_a_mostrar:.1f}"
                cv2.putText(imagem, info_bpm, (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 255, 0), 3)
            else:
                info_bpm = f"O seu BPM está em: {bpm_fixado:.1f}" if bpm_fixado is not None else "Movimento detectado, aguarde..."
                cv2.putText(imagem, info_bpm, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        else:
            # ===================== DEDO REMOVIDO =====================
            cv2.putText(imagem, "Coloque o dedo e pressione 'm' para medir novamente", 
                        (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # ===================== INFORMAÇÕES DE FPS =====================
        cv2.putText(imagem, f"FPS: {taxa_fps:.1f}", (imagem.shape[1] - 200, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # ===================== EXIBIÇÃO DO VÍDEO =====================
    cv2.imshow('Analise via Celular:', imagem)

    # ===================== TECLAS DE CONTROLE =====================
    key = cv2.waitKey(1) & 0xFF
    if key == ord('m'):  # Reiniciar medição
        bpm_fixado = None
        bpm_primeiros = []
        medir_novamente = True
    if key == ord('q'):  # Sair
        break

# ===================== FINALIZAÇÃO =====================
cap.release()
cv2.destroyAllWindows()
