import cv2  # para interagir com a camera
import numpy as np  # faz as contas
import time  # usado para medir o tempo e calcular o fps

Tamanho_tela = 100
# dentro desse parenteses é pra colocar o ip que vai aparecer no aplicativo
url = "http://192.168.100.35:8080/video"
cap = cv2.VideoCapture(url)  # recebe o video
if not cap.isOpened():  # se ele não encontrar camera
    print("Camera não Encontrada")
    exit()

contador_frames = 0
tempo_espera = time.time()
taxa_fps = 30   

buffer_R, buffer_G, buffer_B = [], [], []

valor_bpm = []  # quarda os batimentos por segundo numa lista

print("Coloque seu Dedo na camera do celular")
print("Pressione 'q' na janela de vídeo para sair.")

while True:
    # cap.read retorna dois valores: PRIMEIRO: falso ou verdadeiro(que vai pro retorno) e o SEGUNDO: a imagem (que vai pra imagem)
    retorno, imagem = cap.read()

    if not retorno:
        print("A Stream do video foi perdida, tentando reconectar...")
        cap = cv2.VideoCapture(url)
        time.sleep(2)
        continue
    contador_frames += 1

    if contador_frames % 60 == 0:  # calcula a quantidade de fps a 1 segundo
        cont_final = time.time()  # marca o tempo final
        tempo_decorrido = cont_final - tempo_espera  # marca quanto tempo passou
        if tempo_decorrido > 0:
            taxa_fps = contador_frames/tempo_decorrido  # aqui atualiza a quantidade de fps

            # essas duas linhas zera os contadores
        contador_frames = 0
        tempo_espera = time.time()

    (altura,largura)=imagem.shape[:2]# a variavel shape retorna altura, largura e canais de cores (nessa sequencia), mas como só preciso dos dois primeiros o [:2] pega ate o indice 2 ( mas mas não inclui o indice 2)
    
    tamanho_roi=int(altura * 0.5) #faz o tamanho do ROI ser proporcional a resolução do video, se baseando na altura


    x_inicio=(altura-tamanho_roi)//2 + 380 #fala onde no eixo x o retangulo vai iniciar dividindo a imagem em duas partes iguais 
    y_inicio=(altura-tamanho_roi)//2  #fala onde no eixo y o retangulo vai iniciar
    x_fim=x_inicio+tamanho_roi #calcula onde o ROI vai terminar no eixo x
    y_fim=y_inicio+tamanho_roi#calcula onde o ROI vai terminar no eixo y


   
    
    x_inicio = max(0, x_inicio)#serve pro ROI não ficar pra fora da imagem, se o x_inicio for um numero negativo pra não vazar da imagem o "max" vai escolher o valor 0
    y_inicio = max(0, y_inicio)
    x_fim = min(largura, x_fim)#tambem serve pro ROI não ficar pra fora da imagem, se o x_fim for um numero maior que a largura pra não vazar da imagem o "min" vai escolher o valor da largura
    y_fim = min(altura, y_fim)
    

    if x_fim > x_inicio and y_fim > y_inicio: # serve pra garantir que o ROI seja REALMENTE um retangulo
        roi_central = imagem[y_inicio:y_fim, x_inicio:x_fim] #pega as cordenadas de inicio e fin do ROI tanto do eixo x quanto eixo y

        cv2.rectangle(imagem, (x_inicio, y_inicio), (x_fim, y_fim), (0, 255, 0), 2) # essa linha é a que faz um retangulo verde na imagem
        media_BGR = np.mean(roi_central, axis=(0, 1)) #faz o calculo das cores só de dentro do ROI
    else:
        # Se a ROI for inválida, usa a imagem inteira e avisa
        print("Aviso: ROI inválida, usando imagem inteira.")

        cv2.rectangle(imagem, (0, 0), (largura // 4, altura // 4), (0, 0, 255), 2) # Desenha um quadrado vermelho no canto
        media_BGR = np.mean(imagem, axis=(0, 1)) # deixa de analisar só o retangulo e passa a analisar a iamgem toda


    

    buffer_B.append(media_BGR[0])
    buffer_G.append(media_BGR[1])
    buffer_R.append(media_BGR[2])

    if len(buffer_B) > Tamanho_tela:  # se o tamanho do buffer for maior que o tamanho que foi falado antes então ele tira o mais antigo pra sempre analisar o mais novo

        buffer_B.pop(0)
        buffer_G.pop(0)
        buffer_R.pop(0)

    if len(buffer_B) == Tamanho_tela:

        media_B_recente=media_BGR[0]
        media_G_recente=media_BGR[1]
        media_R_recente=media_BGR[2]

        vermelho=60 # nivel de vermelho minimo

        dedo_na_camera= (media_R_recente > media_G_recente + vermelho and media_R_recente > media_B_recente + vermelho) #confere se as duas contas dão TRUE e retorna pra variavel dedo_na_camera, a variavel "vermelho" quando soma com a media_G_recente fala quanto a media_R_recente tem que ser maior pra ser considerada valida 

        if dedo_na_camera: 
            # normaliza o buffer_R pra não levar em consideração o tom vermelho que mais aparece
            Rnorm = np.array(buffer_R)/(np.mean (buffer_R) + 1e-9)
            Gnorm = np.array(buffer_G)/(np.mean(buffer_G) + 1e-9)
            # normaliza o buffer_G pra não levar em consideração o tom verde que maqqis aparece
            Bnorm = np.array(buffer_B)/(np.mean(buffer_B) + 1e-9)
            # normaliza o buffer_B pra não levar em consideração o tom azul que mais aparece
            S1 = Gnorm-Bnorm  # aqui ele ainda tem o pulso e o ruido
            S2 = -2*Rnorm+Gnorm+Bnorm  # aqui tambem

            # cancelando o ruido de brilho e ampliando o sinal de pulso
            # np.std calcula o desvio padrão(a proporção entre a força dos dois, ou seja o quao o S1 é mais forte que o S2), e depois calcula o fator de ajuste (alpha)
            alpha = np.std(S1)/(np.std(S2) + 1e-9)
            # (alpha*S2) muda o valor do ruido do S2 para se igualar ao ruido do S1, e no final o h é o pulso limpo
            h = S1+(alpha*S2)

            # tira o COMPONENTE DC(np.mean) que é o nivel medio que não varia com o tempo que nesse caso é a cor da pele e a luz ambiente
            entrada_trf = h-np.mean(h)
            # np.fft.fft executa a TRANSFORMADA RAPIDA DE FOURIER que separa separa as frequencias simples que compoe o sinal ( que inclui o bpm)
            trf = np.fft.fft(entrada_trf)


            if taxa_fps > 0:
             # .fftfreq calcula as frequencias da da FFT, Tamanho_tela é o quanto de dados o codigo analisou e d=1.0/taxa_fps é qual o intervalo de tempo entre eles
                frequencias = np.fft.fftfreq(Tamanho_tela, d=1.0/taxa_fps)
            else:
                continue

            # seleciona as frequencias que podem ser batimentos cardiacos
            indices_validos = np.where((frequencias >= 0.67) & (frequencias <= 3))
            # 40 batimentos(pessoa dormindo) / 60 segundos = 0.67
            # 180 batimentos( batimento mais rapido) / 60 segundos = 3

            if len(indices_validos[0]) > 0:
                # np.abs pega o valor absoluto das frequencias exmplo: o valor absoluto de -5 é 5 então ele deixa tudo positivo
                indice_pico = np.argmax(np.abs(trf[indices_validos]))
                # np.argmax guarda a posição do maior valor

                # indices_validos[0][indice_pico] seleciona o maior item de (indices_validos) de acordo com a posição[indice_pico]
                freq_dominante = frequencias[indices_validos[0][indice_pico]]
                # frequencias[indices_validos[0][indice_pico]] pega a maior frequencia de acordo com o indice dado
                bpm_atual = freq_dominante*60  # transforma a frequencia em batimentos
                valor_bpm.append(bpm_atual)  # adiciona o bpm atual na lista

                if len(valor_bpm) > 15:
                    valor_bpm.pop(0)

                media_bpm = np.mean(valor_bpm)

                info_bpm = f"O seu BPM estah em: {media_bpm:.1f}"
                 # coloca as informações no quadro de video
                cv2.putText(imagem, info_bpm, (30, 60),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        else:
            valor_bpm=[]
            cv2.putText(imagem, "Coloque o dedo", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.putText(imagem, f"FPS: {taxa_fps:.1f}", (imagem.shape[1] - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Analise via Celular:', imagem)  # mostra o video na tela do pc

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break

cap.release()

cv2.destroyAllWindows()