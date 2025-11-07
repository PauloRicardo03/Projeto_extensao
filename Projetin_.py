import cv2  # biblioteca pra capturar video
import numpy as np  # faz calculos matematicos
import time  # pra calcular tempo
from scipy.signal import butter, filtfilt, find_peaks
# butter cria tipo um filtro digital que limpa o ruido
# filtfilt aplica um filtro (que tem que criar) no sinal
# find_peaks acha os picos de dados

<<<<<<< HEAD
Tamanho_tela = 100
# dentro desse parenteses é pra colocar o ip que vai aparecer no aplicativo
url = "http://172.20.10.3:8080/video"
cap = cv2.VideoCapture(url)  # recebe o video
if not cap.isOpened():  # se ele não encontrar camera
=======
Tamanho_tela = 100  # analisa os 250 frames
url = "http://192.168.100.49:8080/video"  # ip do aplicativo
cap = cv2.VideoCapture(url)  # pega o sinal da camera
if not cap.isOpened():  # se a camera não abrir
>>>>>>> 159a32ad3603e1ab7467dc85884a71cd6d273631
    print("Camera não Encontrada")
    exit()

contador_frames = 0
<<<<<<< HEAD
tempo_espera = time.time()
=======
tempo_espera = time.time()  # começo de contagem do fps
>>>>>>> 159a32ad3603e1ab7467dc85884a71cd6d273631
taxa_fps = 30

buffer_R = []  # Lista do vermelho

bpm_suavizado = 0.0  # valor do bpm ja suavizado
alpha = 0.05  # usa media exponencial

bpm_para_mostrar = 0.0  # Variável que segura o valor para a tela
tempo_proxima_atualizacao = time.time() + 8.0 # Agenda a primeira atualização para daqui

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

<<<<<<< HEAD
    if contador_frames % 60 == 0:  # calcula a quantidade de fps a 1 segundo
        cont_final = time.time()  # marca o tempo final
        tempo_decorrido = cont_final - tempo_espera  # marca quanto tempo passou
        if tempo_decorrido > 0:
            taxa_fps = (
                contador_frames / tempo_decorrido
            )  # aqui atualiza a quantidade de fps
=======
    contador_frames += 1  # mais um frame processado
    # Calcula quantos segundos desde a ultima medição dos frames
    tempo_decorrido = time.time() - tempo_espera
>>>>>>> 159a32ad3603e1ab7467dc85884a71cd6d273631

    if tempo_decorrido > 1.0:
        taxa_fps = contador_frames / tempo_decorrido  # atualiza a taxa de fps
        contador_frames = 0  # zera o contador
        tempo_espera = time.time()  # zera o tempo de espera

<<<<<<< HEAD
    (largura, altura) = imagem.shape[
        :2
    ]  # a variavel shape retorna altura, largura e canais de cores (nessa sequencia), mas como só preciso dos dois primeiros o [:2] pega ate o indice 2 ( mas mas não inclui o indice 2)

    tamanho_roi = int(
        largura * 0.5
    )  # faz o tamanho do ROI ser proporcional a resolução do video, se baseando na altura

    x_inicio = (
        altura - tamanho_roi
    ) // 2  # fala onde no eixo x o retangulo vai iniciar dividindo a imagem em duas partes iguais
    y_inicio = (
        largura - tamanho_roi
    ) // 2  # fala onde no eixo y o retangulo vai iniciar
    x_fim = x_inicio + tamanho_roi  # calcula onde o ROI vai terminar no eixo x
    y_fim = y_inicio + tamanho_roi  # calcula onde o ROI vai terminar no eixo y

    x_inicio = max(
        0, x_inicio
    )  # serve pro ROI não ficar pra fora da imagem, se o x_inicio for um numero negativo pra não vazar da imagem o "max" vai escolher o valor 0
    y_inicio = max(0, y_inicio)
    x_fim = min(
        altura, x_fim
    )  # tambem serve pro ROI não ficar pra fora da imagem, se o x_fim for um numero maior que a largura pra não vazar da imagem o "min" vai escolher o valor da largura
    y_fim = min(largura, y_fim)

    if (
        x_fim > x_inicio and y_fim > y_inicio
    ):  # serve pra garantir que o ROI seja REALMENTE um retangulo
        roi_central = imagem[
            y_inicio:y_fim, x_inicio:x_fim
        ]  # pega as cordenadas de inicio e fin do ROI tanto do eixo x quanto eixo y

        cv2.rectangle(
            imagem, (x_inicio, y_inicio), (x_fim, y_fim), (0, 255, 0), 2
        )  # essa linha é a que faz um retangulo verde na imagem
        media_BGR = np.mean(
            roi_central, axis=(0, 1)
        )  # faz o calculo das cores só de dentro do ROI
    else:
        # Se a ROI for inválida, usa a imagem inteira e avisa
        print("Aviso: ROI inválida, usando imagem inteira.")

        cv2.rectangle(
            imagem, (0, 0), (largura // 4, altura // 4), (0, 0, 255), 2
        )  # Desenha um quadrado vermelho no canto
        media_BGR = np.mean(
            imagem, axis=(0, 1)
        )  # deixa de analisar só o retangulo e passa a analisar a iamgem toda

    buffer_B.append(media_BGR[0])
    buffer_G.append(media_BGR[1])
    buffer_R.append(media_BGR[2])

    if (
        len(buffer_B) > Tamanho_tela
    ):  # se o tamanho do buffer for maior que o tamanho que foi falado antes então ele tira o mais antigo pra sempre analisar o mais novo

        buffer_B.pop(0)
        buffer_G.pop(0)
        buffer_R.pop(0)

    if len(buffer_B) == Tamanho_tela:

        media_B_recente = media_BGR[0]
        media_G_recente = media_BGR[1]
        media_R_recente = media_BGR[2]

        vermelho = 60  # nivel de vermelho minimo

        dedo_na_camera = (
            media_R_recente > media_G_recente + vermelho
            and media_R_recente > media_B_recente + vermelho
        )  # confere se as duas contas dão TRUE e retorna pra variavel dedo_na_camera, a variavel "vermelho" quando soma com a media_G_recente fala quanto a media_R_recente tem que ser maior pra ser considerada valida

        if dedo_na_camera:
            # normaliza o buffer_R pra não levar em consideração o tom vermelho que mais aparece
            Rnorm = np.array(buffer_R) / (np.mean(buffer_R) + 1e-9)
            Gnorm = np.array(buffer_G) / (np.mean(buffer_G) + 1e-9)
            # normaliza o buffer_G pra não levar em consideração o tom verde que maqqis aparece
            Bnorm = np.array(buffer_B) / (np.mean(buffer_B) + 1e-9)
            # normaliza o buffer_B pra não levar em consideração o tom azul que mais aparece
            S1 = Gnorm - Bnorm  # aqui ele ainda tem o pulso e o ruido
            S2 = -2 * Rnorm + Gnorm + Bnorm  # aqui tambem

            # cancelando o ruido de brilho e ampliando o sinal de pulso
            # np.std calcula o desvio padrão(a proporção entre a força dos dois, ou seja o quao o S1 é mais forte que o S2), e depois calcula o fator de ajuste (alpha)
            alpha = np.std(S1) / (np.std(S2) + 1e-9)
            # (alpha*S2) muda o valor do ruido do S2 para se igualar ao ruido do S1, e no final o h é o pulso limpo
            h = S1 + (alpha * S2)

            # tira o COMPONENTE DC(np.mean) que é o nivel medio que não varia com o tempo que nesse caso é a cor da pele e a luz ambiente
            entrada_trf = h - np.mean(h)
            # np.fft.fft executa a TRANSFORMADA RAPIDA DE FOURIER que separa separa as frequencias simples que compoe o sinal ( que inclui o bpm)
            trf = np.fft.fft(entrada_trf)

            if taxa_fps > 0:
                # .fftfreq calcula as frequencias da da FFT, Tamanho_tela é o quanto de dados o codigo analisou e d=1.0/taxa_fps é qual o intervalo de tempo entre eles
=======
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
        0, y_inicio)  # pra ele não aparecer fora da camera
    x_fim, y_fim = min(largura, x_fim), min(altura, y_fim)  # aqui tambem

    media_R_recente, media_G_recente, media_B_recente = 0, 0, 0

    if x_fim > x_inicio and y_fim > y_inicio:  # ve se o ROI é realmente um retangulo
        # pega só os pixels dentro do ROI
        roi_central = imagem[y_inicio:y_fim, x_inicio:x_fim]
        cv2.rectangle(imagem, (x_inicio, y_inicio), (x_fim, y_fim),
                      (0, 255, 0), 2)  # faz o retangulo na tela
        # faz as medias das cores dentro do ROI
        media_BGR = np.mean(roi_central, axis=(0, 1))
        # põe os valores nas variaveis
        media_B_recente, media_G_recente, media_R_recente = media_BGR[
            0], media_BGR[1], media_BGR[2]
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
        dedo_na_camera = (media_R_recente > media_G_recente + vermelho and
                          # confere se o vermelho é realmente maior que todas as cores mais o nivel
                          media_R_recente > media_B_recente + vermelho)

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
                    sinal_filtrado = filtfilt(
                        b, a, sinal_invertido)  # aplica o filtro
                except Exception:
                    sinal_filtrado = sinal_invertido

            entrada_trf = sinal_filtrado
            # faz a Transformada Rápida de Fourier (FFT) que transforma o sinal vermelho em grafico de frequencias
            trf = np.fft.fft(entrada_trf)

            if taxa_fps > 0:
                # pegas frequencias 1 por 1
>>>>>>> 159a32ad3603e1ab7467dc85884a71cd6d273631
                frequencias = np.fft.fftfreq(Tamanho_tela, d=1.0 / taxa_fps)
            else:
                continue

            indices_validos = np.where((frequencias >= 0.67) & (
                frequencias <= 3.5))  # pega as frequencias dentro da faixa

            if len(indices_validos[0]) > 0:
                # Pega o espectro de frequências válidas
                # pega o valor absoluto das frequencias validas
                sinal_fft = np.abs(trf[indices_validos])
                freqs_validas = frequencias[indices_validos]  # pega os picos

<<<<<<< HEAD
                # indices_validos[0][indice_pico] seleciona o maior item de (indices_validos) de acordo com a posição[indice_pico]
                freq_dominante = frequencias[indices_validos[0][indice_pico]]
                # frequencias[indices_validos[0][indice_pico]] pega a maior frequencia de acordo com o indice dado
                bpm_atual = freq_dominante * 60  # transforma a frequencia em batimentos
                valor_bpm.append(bpm_atual)  # adiciona o bpm atual na lista
=======
                # Encontra TODOS os picos do FFT
                # Pega o pico que tem pelo menos 20% da força do pico máximo
                peaks, properties = find_peaks(sinal_fft, prominence=np.max(
                    sinal_fft) * 0.2)  # Pega os picos do grafico FFT
>>>>>>> 159a32ad3603e1ab7467dc85884a71cd6d273631

                if len(peaks) > 0:
                    # Ordena os picos do mais forte para o mais fraco
                    sorted_peak_indices = np.argsort(sinal_fft[peaks])[
                        ::-1]  # faz o sort do maior pro menor

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
                            bpm_suavizado = (bpm_atual * alpha) + \
                                (bpm_suavizado * (1.0 - alpha))

<<<<<<< HEAD
                info_bpm = f"O seu BPM estah em: {media_bpm:.1f}"
                # coloca as informações no quadro de video
                cv2.putText(
                    imagem,
                    info_bpm,
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 0),
                    3,
                )

        else:
            valor_bpm = []
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

    cv2.imshow("Analise via Celular:", imagem)  # mostra o video na tela do pc

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break

cap.release()

=======
        else:

            bpm_suavizado = 0.0
    agora = time.time()
    if agora > tempo_proxima_atualizacao:
        # Já se passaram 5 segundos? Hora de atualizar o que o usuário vê.
        
        if bpm_suavizado > 0: # Se o cálculo for válido
            bpm_para_mostrar = bpm_suavizado # Atualiza o display
        else:
            bpm_para_mostrar = 0.0 # Reseta o display se o dedo for removido
        
        # Agenda a próxima atualização para daqui a 5 segundos
        tempo_proxima_atualizacao = agora + 5.0
    if bpm_para_mostrar > 0:
        info_bpm = f"Batimentos por Minuto: {bpm_para_mostrar:.1f}"
        cv2.putText(imagem, info_bpm, (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    else:
        cv2.putText(imagem, "Coloque o dedo", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2) 
    cv2.putText(imagem, f"FPS: {taxa_fps:.1f}", (
        imagem.shape[1] - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Analise via Celular:', imagem)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
>>>>>>> 159a32ad3603e1ab7467dc85884a71cd6d273631
cv2.destroyAllWindows()
