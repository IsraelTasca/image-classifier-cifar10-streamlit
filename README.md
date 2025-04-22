
# Classificador de Imagens CIFAR-10 com Streamlit 🚀

Este projeto é um app de classificação de imagens usando uma Rede Neural Convolucional (CNN) treinada no famoso dataset **CIFAR-10**.  
O app foi desenvolvido em **Python** usando **Streamlit** para criar uma interface interativa de fácil uso.

## Demonstração

O app permite:

- Treinar uma CNN com dados do CIFAR-10
- Visualizar imagens de amostra do dataset
- Fazer upload de uma imagem `.jpg` ou `.png`
- Classificar a imagem enviada
- Mostrar a confiança do modelo para cada classe

## Dataset 

Utilizamos o **CIFAR-10**, que contém 60.000 imagens coloridas 32x32 divididas em 10 classes:

- Avião 
- Automóvel 
- Pássaro 
- Gato 
- Veado 
- Cachorro 
- Sapo 
- Cavalo 
- Navio 
- Caminhão 

## Como rodar o projeto localmente 🛠️

### 1. Clone o repositório

```bash
git clone https://github.com/IsraelTasca/image-classifier-cifar10-streamlit.git
cd image-classifier-cifar10-streamlit
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Rode o app

```bash
streamlit run app.py
```
