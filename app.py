import streamlit as st
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers
import matplotlib.pyplot as plt
import numpy as np
import tarfile
import os
import ssl
from PIL import Image

ssl._create_default_https_context = ssl._create_unverified_context

#Função para carregar e preparar os dados
@st.cache_data
def load_data():
    local_path = "C:/Users/Israel Tasca/Desktop/classificador-cifar10/cifar-10-python.tar.gz"

    if not os.path.exists("cifar-10-batches-py"):
        with tarfile.open(local_path, "r:gz") as tar:
            tar.extractall()

    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    return train_images, train_labels, test_images, test_labels

#Função para construir o modelo CNN
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.Dense(10)
    ])
    return model

st.title("📸 Classificador de Imagens CIFAR-10 com CNN")
st.write("Este app treina uma rede neural convolucional para classificar imagens do conjunto CIFAR-10.")

train_images, train_labels, test_images, test_labels = load_data()

if st.checkbox("Mostrar imagens de amostra"):
    class_names = ['avião', 'automóvel', 'pássaro', 'gato', 'veado',
                   'cachorro', 'sapo', 'cavalo', 'navio', 'caminhão']
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(train_images[i])
        ax.set_title(class_names[train_labels[i][0]])
        ax.axis('off')
    st.pyplot(fig)


if "model" not in st.session_state:
    st.session_state.model = None

#Botão para treinar o modelo
if st.button("🔁 Treinar Modelo"):
    with st.spinner("Inicializando o treinamento..."):
        model = build_model()
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

       
        progress_bar = st.progress(0)
        status_text = st.empty()

        epochs = 10
        history = {"accuracy": [], "val_accuracy": []}

        for epoch in range(epochs):
            status_text.text(f"🧠 Treinando época {epoch + 1}/{epochs}")
            hist = model.fit(train_images, train_labels, epochs=1,
                             validation_data=(test_images, test_labels), verbose=0)

            history["accuracy"].append(hist.history['accuracy'][0])
            history["val_accuracy"].append(hist.history['val_accuracy'][0])

            progress_bar.progress((epoch + 1) / epochs)

        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
        st.success(f"✅ Treinamento concluído! Acurácia no teste: {test_acc:.4f}")

        fig_acc = plt.figure()
        plt.plot(history["accuracy"], label='Treino')
        plt.plot(history["val_accuracy"], label='Validação')
        plt.xlabel('Época')
        plt.ylabel('Acurácia')
        plt.title('Acurácia ao longo das épocas')
        plt.legend()
        st.pyplot(fig_acc)

        st.session_state.model = model  


st.header("📤 Testar Modelo com uma Imagem")

uploaded_file = st.file_uploader("Envie uma imagem .jpg ou .png", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((32, 32))
    st.image(image, caption='Imagem enviada', use_container_width=True)

    img_array = np.array(image) / 255.0
    img_array = img_array.reshape((1, 32, 32, 3))

    if st.session_state.model is None:
        st.warning("⚠️ Você precisa treinar o modelo antes de testar uma imagem.")
    else:
        with st.spinner("🔍 Analisando a imagem..."):
            predictions = st.session_state.model.predict(img_array)
            predicted_class = np.argmax(predictions)
            confidence = tf.nn.softmax(predictions[0])

            class_names = ['avião', 'automóvel', 'pássaro', 'gato', 'veado',
                           'cachorro', 'sapo', 'cavalo', 'navio', 'caminhão']

            st.subheader(f"📌 Predição: **{class_names[predicted_class]}**")

            st.subheader("📊 Confiança nas classes:")
            fig_conf = plt.figure(figsize=(10, 4))
            plt.bar(class_names, confidence)
            plt.xticks(rotation=45)
            plt.ylabel("Confiança")
            plt.ylim([0, 1])
            st.pyplot(fig_conf)



