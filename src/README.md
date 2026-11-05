# 💓 Monitor Multiparamétrico PPG & rPPG em Tempo Real (Web & Upload)

Este projeto é uma aplicação web desenvolvida em **Python** utilizando **Streamlit** que transforma a câmera de um dispositivo (smartphone ou webcam) em um monitor multiparamétrico de sinais vitais.

O sistema utiliza tanto a técnica de **Fotopletismografia de Contato (PPG)** (analisando as variações na absorção de luz nos canais vermelho e azul na ponta do dedo) quanto **Fotopletismografia Remota (rPPG)** (capturando variações sutis no canal verde refletido pela região da testa sob luz ambiente).

---

## 📋 Funcionalidades

O projeto opera em três modos distintos:

1. **📷 Modo Dedo — Traseira + Flash (PPG de Contato: BPM + SpO2)**
   * Utiliza **WebRTC** com servidor STUN configurado para streaming de vídeo direto no navegador (local ou via túnel HTTPS no celular).
   * Processamento assíncrono (**Multithreading** + `threading.Lock`) para não travar a renderização de vídeo.
   * Utiliza a lanterna (`torch`) do dispositivo móvel para iluminar o tecido capilar do dedo.
   * **Cálculo de BPM:** Reamostragem uniforme ($30\text{ Hz}$), filtro passa-faixa Butterworth de 3ª ordem ($0.85\text{ Hz}$ a $2.5\text{ Hz}$), Transformada Rápida de Fourier (`rfft`) com janelamento de Hann e Zero-Padding ($N=2048$), validado por SNR espectral e consenso de picos (`find_peaks`).
   * **Cálculo de Oximetria ($\text{SpO}_2\%$):** Método da Razão de Razões ($R_{os}$) entre componentes pulsáteis ($AC$) e basais ($DC$) dos canais Vermelho e Azul.
   * Gráfico contínuo em tempo real via `add_rows` (sem piscar a interface).

2. **🤳 Modo Rosto — Frontal (rPPG Sem Contato: BPM + RPM)**
   * Detecção facial automática via **Haar Cascade** (`OpenCV`) com suavização espacial e isolamento da **ROI da testa**.
   * Extração do canal **Verde (G)** em janela deslizante de até 300 amostras (~10s).
   * **Frequência Cardíaca (BPM):** Filtragem passa-faixa ($0.85\text{ Hz}$ a $2.4\text{ Hz}$) e análise espectral + temporal.
   * **Frequência Respiratória (RPM):** Filtragem passa-faixa de baixa frequência ($0.15\text{ Hz}$ a $0.40\text{ Hz}$, correspondente a 9–24 respirações/minuto).

3. **📂 Modo Vídeo (Upload Offline)**
   * Permite o envio de arquivos de vídeo pré-gravados (`.mp4`, `.mov`, `.avi`, `.mkv`).
   * Processamento frame a frame com extração simultânea de **BPM** e **SpO2 (%)** estabilizados.

---

## 🛠️ Tecnologias Utilizadas

* **[Streamlit](https://streamlit.io/):** Interface web interativa e dashboards em tempo real.
* **[Streamlit-WebRTC](https://github.com/whitphx/streamlit-webrtc):** Captura de câmera e transmissão WebRTC no navegador.
* **[OpenCV (cv2)](https://opencv.org/):** Processamento de imagem, recorte de ROI e detecção facial.
* **[NumPy](https://numpy.org/) & [Pandas](https://pandas.pydata.org/):** Computação vetorial, interpolação temporal e séries de dados.
* **[SciPy](https://scipy.org/):** Processamento digital de sinais (`butter`, `filtfilt`, `detrend`, `rfft`, `rfftfreq`, `find_peaks`).
* **[PyAV (av)](https://github.com/PyAV-Org/PyAV):** Manipulação de frames de vídeo WebRTC.

---

## 🚀 Como Executar o Projeto

### Pré-requisitos

Certifique-se de ter o **Python 3.8+** instalado.

### 1. Clonar o repositório

```bash
git clone https://github.com/PauloRicardo03/Projeto_extensao.git
cd Projeto_extensao
```

### 2. Instalar as dependências

```bash
pip install streamlit streamlit-webrtc opencv-python numpy pandas scipy av
```

### 3. Executar a aplicação

```bash
streamlit run src/bpm_extractor.py --server.enableCORS=false --server.enableXsrfProtection=false
```

O navegador abrirá automaticamente no endereço `http://localhost:8501`. Para testar no celular via HTTPS, utilize um túnel como `npx cloudflared tunnel --url http://localhost:8501`.

---

## 👥 Autores

Este projeto foi desenvolvido pela equipe de discentes do curso de Sistemas de Informação:

* **[Amanda Gabriela Araújo Costa](https://www.linkedin.com/in/amanda-gabriela-9ab83b21b/)**
* **[Kenzo Ramos Otaguiri](https://www.linkedin.com/in/kenzo-otaguiri-b72720234/)**
* **[Mariana Khodr Susini](https://www.linkedin.com/in/mariana-khodr-susini/)**
* **[Paulo Ricardo Ferreira Lacerda](https://www.linkedin.com/in/paulo-ricardo-ferreira-lacerda-918694381/)**

---

## 👨‍🏫 Orientação Acadêmica

Todo o desenvolvimento e aplicação dos conceitos matemáticos e de processamento de sinais foram supervisionados por:

* **[Prof. Maxwell Gomes](https://www.linkedin.com/in/mxyconsulting/)** – *Orientador do Projeto*

---

## 📄 Licença

Este projeto está sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.
