# 💓 Monitor Cardíaco PPG em Tempo Real (Web & Upload)

Este projeto é uma aplicação web desenvolvida em **Python** utilizando **Streamlit** que transforma a câmera de um dispositivo (smartphone ou webcam) em um monitor cardíaco.

O sistema utiliza a técnica de **Fotopletismografia (PPG)** remota, analisando as variações sutis na absorção de luz (canal vermelho) causadas pelo fluxo sanguíneo na ponta do dedo.

---

## 📋 Funcionalidades

O projeto opera em dois modos distintos:

1.  **📷 Modo Câmera (Tempo Real)**
    * Utiliza **WebRTC** para streaming de vídeo direto no navegador.
    * Processamento assíncrono (Multithreading) para não travar a interface.
    * Utiliza a lanterna (flash) do dispositivo para iluminar o dedo.
    * Cálculo de BPM via **Transformada de Fourier (FFT)** para maior precisão espectral.
    * Gráfico de histórico de batimentos em tempo real.

2.  **📂 Modo Vídeo (Upload)**
    * Permite o envio de arquivos de vídeo pré-gravados (.mp4, .mov, etc).
    * Processamento frame a frame com **OpenCV**.
    * Cálculo de BPM via **Detecção de Picos (Time-Domain)**.

---

## 🛠️ Tecnologias Utilizadas

O projeto foi construído utilizando as seguintes bibliotecas:

* **[Streamlit](https://streamlit.io/):** Interface web interativa e dashboards.
* **[Streamlit-WebRTC](https://github.com/whitphx/streamlit-webrtc):** Conexão da câmera do navegador com o backend Python.
* **[OpenCV (cv2)](https://opencv.org/):** Processamento de imagem e manipulação de vídeo.
* **[NumPy](https://numpy.org/):** Cálculos matemáticos e manipulação de arrays de alta performance.
* **[SciPy](https://scipy.org/):** Processamento de sinais (Filtros Butterworth, FFT e Find Peaks).
* **[Matplotlib](https://matplotlib.org/):** (Opcional) Para plotagem de gráficos avançados.
* **[Av](https://github.com/PyAV-Org/PyAV):** Manipulação de containers de vídeo e áudio.

---

## 🚀 Como Executar o Projeto

### Pré-requisitos

Certifique-se de ter o **Python 3.8+** instalado.

### 1. Clonar o repositório

```bash
git clone [https://github.com/PauloRicardo03/Projeto_extensao.git](https://github.com/PauloRicardo03/Projeto_extensao.git)
cd Projeto_extensao
````

### 2\. Instalar as dependências

Recomenda-se criar um ambiente virtual antes da instalação:

```bash
pip install -r requirements.txt
```

### 3\. Executar a aplicação

```bash
streamlit run app.py
```

O navegador abrirá automaticamente no endereço `http://localhost:8501`.

-----

## 👥 Autores

Este projeto foi desenvolvido pela equipe de discentes do curso de Sistemas de Informação:

 * **[Amanda Gabriela Araújo Costa](https://www.linkedin.com/in/amanda-gabriela-9ab83b21b/)**
* **[Kenzo Ramos Otaguiri](https://www.linkedin.com/in/kenzo-otaguiri-b72720234/)**
* **[Mariana Khodr Susini](https://www.linkedin.com/in/mariana-khodr-susini/)**
* **[Paulo Ricardo Ferreira Lacerda](https://www.linkedin.com/in/paulo-ricardo-ferreira-lacerda-918694381/)**

-----

## 👨‍🏫 Orientação Acadêmica

Todo o desenvolvimento e aplicação dos conceitos matemáticos e de processamento de sinais foram supervisionados por:

  * **[Prof. Maxwell Gomes](https://www.linkedin.com/in/mxyconsulting/)** – *Orientador do Projeto*

-----

## 📄 Licença

Este projeto está sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

```
```