import cv2 
import numpy as np  
import time  # Para medir o tempo e calcular FPS
from scipy.signal import (
    butter,
    filtfilt,
    find_peaks,
)  # Para filtragem e detecção de picos

# ===================== CONFIGURAÇÕES =====================
Tamanho_tela = 100  # Quantidade de frames que vamos analisar por vez
<<<<<<< HEAD
url = "http://172.20.10.3:8080/video"  # IP da câmera no aplicativo
cap = cv2.VideoCapture(url)  # Captura o vídeo via IP
=======
url = "http://192.168.100.182:4747/video"  # IP da camera no aplicativo
cap = cv2.VideoCapture(url)  # pega o video da camera
>>>>>>> 159a32ad3603e1ab7467dc85884a71cd6d273631

if not cap.isOpened():  # verifica se abriu a camera mesmo
    print("Camera não Encontrada")
    exit()

contador_frames = 0
tempo_espera = time.time()#começa a contar o tempo
taxa_fps = 30 #só pra exemplificar pq vai mudar esse valor


<<<<<<< HEAD
# Variáveis para fixar BPM
bpm_fixado = None  # Para armazenar o BPM fixo
bpm_primeiros = []  # Para armazenar os primeiros 2-3 BPM detectados
medir_novamente = True  # Controla se deve medir novos batimentos
=======
buffer_R, buffer_G, buffer_B = [], [], [] #armazena cada cor em listas


bpm_fixado = None       # aramazena o bpm fixo
bpm_primeiros = []      # armazena os primeiros bpm
medir_novamente = True  # variavel pra medir os batimentos denovo
>>>>>>> 159a32ad3603e1ab7467dc85884a71cd6d273631

print("Coloque seu Dedo na camera do celular")
print("Pressione 'q' na janela de vídeo para sair.")

<<<<<<< HEAD

# ===================== FUNÇÃO DE FILTRAGEM =====================
def bandpass_filter(signal, low=0.8, high=3, fs=30):
    nyq = 0.5 * fs
    low /= nyq
    high /= nyq
    b, a = butter(2, [low, high], btype="band")
    return filtfilt(b, a, signal)


# ===================== LOOP PRINCIPAL =====================
=======
#48(quando a pessoa ta dormindo)/60=0.8
#180(quando ta fazendo exercicio)/60=3
def bandpass_filter(signal, low=0.8, high=3, fs=30):
    fr_nyquist = 0.5 * fs #clcula a frequencia maxima que da pra capturar com precisão
    low = low / fr_nyquist#normalizando a menor frequencia
    high = high / fr_nyquist#normaliza a frequencia maior
    b, a = butter(2, [low, high], btype='band')#btype='band' deixa passr só oque esta entre o low e o high
    return filtfilt(b, a, signal)# aplica o fltro no signal


>>>>>>> 159a32ad3603e1ab7467dc85884a71cd6d273631
while True:
    retorno, imagem = cap.read()  # Captura um frame da câmera

    if not retorno:# verifica se recebeu as imagens
        print("A Stream do video foi perdida, tentando reconectar...")
        cap = cv2.VideoCapture(url)
        time.sleep(2)# tempo pra tentar reconcectar
        continue

    contador_frames += 1

    
    if contador_frames % 60 == 0:
        cont_final = time.time()#pega o tempo final
        tempo_decorrido = cont_final - tempo_espera#calcula quanto tempo passou
        if tempo_decorrido > 0:
            taxa_fps = contador_frames / tempo_decorrido #calculo de fps
        #daqui pra baixo ele reseta os valores denovo
        contador_frames = 0
        tempo_espera = time.time()

   
    (largura, altura) = imagem.shape[:2]# a variavel shape retorna altura, largura e canais de cores (nessa sequencia), mas como só preciso dos dois primeiros o [:2] pega ate o indice 2 ( mas mas não inclui o indice 2)
    tamanho_roi = int(largura * 0.5)#faz o tamanho do ROI ser proporcional a resolução do video, se baseando na altura

    x_inicio = (altura - tamanho_roi) // 2#fala onde no eixo x o retangulo vai iniciar dividindo a imagem em duas partes iguais 
    y_inicio = (largura - tamanho_roi) // 2#fala onde no eixo y o retangulo vai iniciar
    x_fim = x_inicio + tamanho_roi#calcula onde o ROI vai terminar no eixo x
    y_fim = y_inicio + tamanho_roi#calcula onde o ROI vai terminar no eixo y

    x_inicio = max(0, x_inicio)#serve pro ROI não ficar pra fora da imagem, se o x_inicio for um numero negativo pra não vazar da imagem o "max" vai escolher o valor 0
    y_inicio = max(0, y_inicio)
    x_fim = min(altura, x_fim)#tambem serve pro ROI não ficar pra fora da imagem, se o x_fim for um numero maior que a largura pra não vazar da imagem o "min" vai escolher o valor da largura
    y_fim = min(largura, y_fim)

   
    if x_fim > x_inicio and y_fim > y_inicio:# serve pra garantir que o ROI seja REALMENTE um retangulo
        roi_central = imagem[y_inicio:y_fim, x_inicio:x_fim]#pega as cordenadas de inicio e fin do ROI tanto do eixo x quanto eixo y
        cv2.rectangle(imagem, (x_inicio, y_inicio), (x_fim, y_fim), (0, 255, 0), 2)# essa linha é a que faz um retangulo verde na imagem
        media_BGR = np.mean(roi_central, axis=(0, 1))#faz o calculo das cores só de dentro do ROI
    else:# Se a ROI for inválida, usa a imagem inteira e avisa
        cv2.rectangle(imagem, (0, 0), (largura // 4, altura // 4), (0, 0, 255), 2)
        media_BGR = np.mean(imagem, axis=(0, 1))

    
    buffer_B.append(media_BGR[0])
    buffer_G.append(media_BGR[1])
    buffer_R.append(media_BGR[2])

    if len(buffer_B) > Tamanho_tela:# se o tamanho do buffer for maior que o tamanho que foi falado antes então ele tira o mais antigo pra sempre analisar o mais novo
        buffer_B.pop(0)
        buffer_G.pop(0)
        buffer_R.pop(0)

    
    if len(buffer_B) == Tamanho_tela:
        media_B_recente = media_BGR[0]
        media_G_recente = media_BGR[1]
        media_R_recente = media_BGR[2]

        vermelho = 110  # nível mínimo de vermelho

<<<<<<< HEAD
        dedo_na_camera = (
            media_R_recente > media_G_recente + vermelho
            and media_R_recente > media_B_recente + vermelho
        )
=======
        dedo_na_camera = (media_R_recente > media_G_recente + vermelho and media_R_recente > media_B_recente + vermelho)#confere se as duas contas dão TRUE e retorna pra variavel dedo_na_camera, a variavel "vermelho" quando soma com a media_G_recente fala quanto a media_R_recente tem que ser maior pra ser considerada valida 

       #if dedo_na_camera and medir_novamente:
            # Gnorm = np.array(buffer_G) / (np.mean(buffer_G) + 1e-9)
            # sinal_ac = Gnorm - np.mean(Gnorm)
            # sinal_filtrado = bandpass_filter(sinal_ac, low=0.8, high=3, fs=taxa_fps)
            # peaks, _ = find_peaks(sinal_filtrado, distance=taxa_fps*0.4)
>>>>>>> 159a32ad3603e1ab7467dc85884a71cd6d273631

        if dedo_na_camera and medir_novamente:
            Rnorm = np.array(buffer_R) / (np.mean(buffer_R) + 1e-9)
            sinal_ac = -(Rnorm - np.mean(Rnorm))  # usa canal vermelho e inverte
            sinal_filtrado = bandpass_filter(sinal_ac, low=0.8, high=3, fs=taxa_fps)
            peaks, _ = find_peaks(sinal_filtrado, distance=taxa_fps * 0.4)

            if len(peaks) > 1:
                intervalos = np.diff(peaks) / taxa_fps# calcula os intervalos entre os picos de  frequencia
                bpm_atual = 60 / np.mean(intervalos)#transforma em bpm

                
                if bpm_fixado is None:
                    #começa a colocar os bpm na lista
                    bpm_primeiros.append(bpm_atual)
                    if len(bpm_primeiros) >= 3:#só calcula a media depois de 3bpm
                        bpm_fixado = np.mean(bpm_primeiros)#poe a media em bpm_fixado

<<<<<<< HEAD
                valor_a_mostrar = bpm_fixado if bpm_fixado is not None else bpm_atual
                info_bpm = f"O seu BPM esta em: {valor_a_mostrar:.1f}"
                cv2.putText(
                    imagem,
                    info_bpm,
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 0),
                    3,
                )
=======
                valor_a_mostrar = bpm_fixado if bpm_fixado is not None else bpm_atual # ve se ja tem bpm_fixado se não tiver ele usa o atual mesmo
                info_bpm = f"O seu BPM está em: {valor_a_mostrar:.1f}"
                cv2.putText(imagem, info_bpm, (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 255, 0), 3)
>>>>>>> 159a32ad3603e1ab7467dc85884a71cd6d273631
            else:
                info_bpm = (
                    f"O seu BPM estah em: {bpm_fixado:.1f}"
                    if bpm_fixado is not None
                    else "Movimento detectado, aguarde..."
                )
                cv2.putText(
                    imagem,
                    info_bpm,
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )

        else:
<<<<<<< HEAD
            # ===================== DEDO REMOVIDO =====================
            cv2.putText(
                imagem,
                "Coloque o dedo e pressione 'm' para medir novamente",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )

        # ===================== INFORMAÇÕES DE FPS =====================
        cv2.putText(
            imagem,
            f"FPS: {taxa_fps:.1f}",
            (imagem.shape[1] - 200, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )

    # ===================== EXIBIÇÃO DO VÍDEO =====================
    cv2.imshow("Analise via Celular:", imagem)
=======
            
            cv2.putText(imagem, "Coloque o dedo e pressione 'm' para medir novamente", 
                        (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        
        cv2.putText(imagem, f"FPS: {taxa_fps:.1f}", (imagem.shape[1] - 200, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    
    cv2.imshow('Analise via Celular:', imagem)
>>>>>>> 159a32ad3603e1ab7467dc85884a71cd6d273631

    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("m"):  # Reiniciar medição
        bpm_fixado = None
        bpm_primeiros = []
        medir_novamente = True
    if key == ord("q"):  # Sair
        break


cap.release()
cv2.destroyAllWindows()

