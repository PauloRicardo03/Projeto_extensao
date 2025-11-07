import cv2  # biblioteca pra capturar video
import numpy as np  # faz calculos matematicos
import time  # pra calcular tempo
from scipy.signal import butter, filtfilt, find_peaks

# butter cria tipo um filtro digital que limpa o ruido
# filtfilt aplica um filtro (que tem que criar) no sinal
# find_peaks acha os picos de dados

Tamanho_tela = 100  # analisa os 250 frames
url = "http://192.168.100.49:8080/video"  # ip do aplicativo
cap = cv2.VideoCapture(url)  # pega o sinal da camera
if not cap.isOpened():  # se a camera não abrir
    print("Camera não Encontrada")
    exit()

contador_frames = 0
tempo_espera = time.time()  # começo de contagem do fps
taxa_fps = 30

buffer_R = []  # Lista do vermelho

bpm_suavizado = 0.0  # valor do bpm ja suavizado
alpha = 0.05  # usa media exponencial

bpm_para_mostrar = 0.0  # Variável que segura o valor para a tela
tempo_proxima_atualizacao = (
    time.time() + 8.0
)  # Agenda a primeira atualização para daqui

print("Coloque seu Dedo na camera do celular")
print("Pressione 'q' na janela de vídeo para sair.")

while True:
    # primeiro valor é se conseguiu captar a imagem, e o segundo é a imagem
    retorno, imagem = cap.read()
    if not retorno:  # se a camera cair
        print("Stream do video foi perdida, tentando reconectar")
        cap = cv2.VideoCapture(url)  # tenta reconectar
        time.sleep(2)  # tenta de 2 em 2 segundos
        continue

    contador_frames += 1  # mais um frame processado
    # Calcula quantos segundos desde a ultima medição dos frames
    tempo_decorrido = time.time() - tempo_espera

    if tempo_decorrido > 1.0:
        taxa_fps = contador_frames / tempo_decorrido  # atualiza a taxa de fps
        contador_frames = 0  # zera o contador
        tempo_espera = time.time()  # zera o tempo de espera

    (altura, largura) = imagem.shape[:2]  # pega as dimensões
    # faz o roi ter metade do tamanho do menor lado da imagem
    tamanho_roi = int(min(altura, largura) * 0.5)
    # coordenada do inicio x (superior esquerdo)
    x_inicio = (largura - tamanho_roi) // 2
    # coordenada do inicio y (superior esquerdo)
    y_inicio = (altura - tamanho_roi) // 2
    # coordenada do fim x (canto superior esquerdo)
    x_fim = x_inicio + tamanho_roi
    # coordenada do fim y (canto superior esquerdo)
    y_fim = y_inicio + tamanho_roi
    x_inicio, y_inicio = max(0, x_inicio), max(
        0, y_inicio
    )  # pra ele não aparecer fora da camera
    x_fim, y_fim = min(largura, x_fim), min(altura, y_fim)  # aqui tambem

    media_R_recente, media_G_recente, media_B_recente = 0, 0, 0

    if x_fim > x_inicio and y_fim > y_inicio:  # ve se o ROI é realmente um retangulo
        # pega só os pixels dentro do ROI
        roi_central = imagem[y_inicio:y_fim, x_inicio:x_fim]
        cv2.rectangle(
            imagem, (x_inicio, y_inicio), (x_fim, y_fim), (0, 255, 0), 2
        )  # faz o retangulo na tela
        # faz as medias das cores dentro do ROI
        media_BGR = np.mean(roi_central, axis=(0, 1))
        # põe os valores nas variaveis
        media_B_recente, media_G_recente, media_R_recente = (
            media_BGR[0],
            media_BGR[1],
            media_BGR[2],
        )
    else:  # se não for um retangulo ele pega a imagem inteira mesmo

        media_BGR = np.mean(imagem, axis=(0, 1))
        media_B_recente = media_BGR[0]
        media_G_recente = media_BGR[1]
        media_R_recente = media_BGR[2]

    buffer_R.append(media_R_recente)  # poe o valor no final da lista
    if len(buffer_R) > Tamanho_tela:  # mantem a lista sempre atualizada
        buffer_R.pop(0)  # tira o mais antigo

    if len(buffer_R) == Tamanho_tela:
        vermelho = 60  # nivel pra conferir se o dedo esta realmente na tela
        dedo_na_camera = (
            media_R_recente > media_G_recente + vermelho
            and
            # confere se o vermelho é realmente maior que todas as cores mais o nivel
            media_R_recente > media_B_recente + vermelho
        )

        if dedo_na_camera:
            # transforma em um array numpy pra fazer calculos
            R_array = np.array(buffer_R)
            # normaliza o sinal vermelho
            Rnorm = R_array / (np.mean(R_array) + 1e-9)
            # tira os valores que mais aparecem que seria a cor da pele e a luz
            sinal_ac = Rnorm - np.mean(Rnorm)
            sinal_invertido = -sinal_ac  # inverte o sinal

            sinal_filtrado = sinal_invertido
            if taxa_fps > 0:
                try:
                    fs = taxa_fps  # frequencia de amostragem
                    nyq = 0.5 * fs  # calcula a frequencia de Nyquist
                    # define os batimentos minimos(07*60=42bpm(dormindo)) e os batimentos maximos(3.5*60=210bpm(exercitando))
                    low, high = 0.7 / nyq, 3.5 / nyq
                    # cria o filtro
                    b, a = butter(2, [low, high], btype="band")
                    sinal_filtrado = filtfilt(b, a, sinal_invertido)  # aplica o filtro
                except Exception:
                    sinal_filtrado = sinal_invertido

            entrada_trf = sinal_filtrado
            # faz a Transformada Rápida de Fourier (FFT) que transforma o sinal vermelho em grafico de frequencias
            trf = np.fft.fft(entrada_trf)

            if taxa_fps > 0:
                # pegas frequencias 1 por 1
                frequencias = np.fft.fftfreq(Tamanho_tela, d=1.0 / taxa_fps)
            else:
                continue

            indices_validos = np.where(
                (frequencias >= 0.67) & (frequencias <= 3.5)
            )  # pega as frequencias dentro da faixa

            if len(indices_validos[0]) > 0:
                # Pega o espectro de frequências válidas
                # pega o valor absoluto das frequencias validas
                sinal_fft = np.abs(trf[indices_validos])
                freqs_validas = frequencias[indices_validos]  # pega os picos

                # Encontra TODOS os picos do FFT
                # Pega o pico que tem pelo menos 20% da força do pico máximo
                peaks, properties = find_peaks(
                    sinal_fft, prominence=np.max(sinal_fft) * 0.2
                )  # Pega os picos do grafico FFT

                if len(peaks) > 0:
                    # Ordena os picos do mais forte para o mais fraco
                    sorted_peak_indices = np.argsort(sinal_fft[peaks])[
                        ::-1
                    ]  # faz o sort do maior pro menor

                    # Pega o índice do pico mais forte
                    strongest_peak_fft_index = peaks[sorted_peak_indices[0]]

                    # faz o calculo do BPM com o pico mais forte
                    bpm_atual = freqs_validas[strongest_peak_fft_index] * 60

                    confiante = True
                    if len(peaks) > 1:
                        # Se houver um segundo pico, verifica a confiança
                        # pega o segundo pico mais forte
                        second_strongest_fft_index = peaks[sorted_peak_indices[1]]

                        strongest_amplitude = sinal_fft[strongest_peak_fft_index]
                        # pega a altura dos dois picos
                        second_amplitude = sinal_fft[second_strongest_fft_index]

                        # TRAVA: Se o pico mais forte não for 2x mais forte que o segundo,
                        # o sinal é ambíguo. Não atualize.
                        # se o pico 1 não for 2.5 vezes maior que o segundo então quer dizer que o sinal tem ruido
                        if strongest_amplitude < second_amplitude * 2.5:
                            confiante = False

                    if confiante:
                        # SÓ ATUALIZA SE TIVER VALIDADO
                        if bpm_suavizado == 0.0:  # ve se é a primeira leitura
                            bpm_suavizado = bpm_atual  # se for ja pega o valor atual
                        else:
                            # suaviza o bpm
                            bpm_suavizado = (bpm_atual * alpha) + (
                                bpm_suavizado * (1.0 - alpha)
                            )

        else:

            bpm_suavizado = 0.0
    agora = time.time()
    if agora > tempo_proxima_atualizacao:
        # Já se passaram 5 segundos? Hora de atualizar o que o usuário vê.

        if bpm_suavizado > 0:  # Se o cálculo for válido
            bpm_para_mostrar = bpm_suavizado  # Atualiza o display
        else:
            bpm_para_mostrar = 0.0  # Reseta o display se o dedo for removido

        # Agenda a próxima atualização para daqui a 5 segundos
        tempo_proxima_atualizacao = agora + 5.0
    if bpm_para_mostrar > 0:
        info_bpm = f"Batimentos por Minuto: {bpm_para_mostrar:.1f}"
        cv2.putText(
            imagem, info_bpm, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3
        )
    else:
        cv2.putText(
            imagem,
            "Coloque o dedo",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )
    cv2.putText(
        imagem,
        f"FPS: {taxa_fps:.1f}",
        (imagem.shape[1] - 200, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )
    cv2.imshow("Analise via Celular:", imagem)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
