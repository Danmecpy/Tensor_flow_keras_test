import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import numpy as np
from PIL import Image, ImageOps

# Carregar dados MNIST e pré-processar
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalizar

# Construir o modelo
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train[..., np.newaxis], y_train, epochs=3, validation_split=0.1)

# Interface com Streamlit
st.title("Reconhecimento de Dígitos com IA")
st.write("Desenhe um dígito (0 a 9) e a IA tentará reconhecer!")

# Criar área de upload
uploaded_file = st.file_uploader("Envie uma imagem em preto e branco (28x28 pixels)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Carregar imagem, converter para escala de cinza e redimensionar
    image = Image.open(uploaded_file).convert("L")
    image = ImageOps.invert(image)  # Inverter cores: fundo preto, dígito branco
    image = image.resize((28, 28))
    img_array = np.array(image) / 255.0  # Normalizar

    # Mostrar imagem
    st.image(image, caption="Imagem Carregada", use_column_width=True)

    # Previsão do modelo
    pred = model.predict(img_array.reshape(1, 28, 28, 1))
    pred_label = np.argmax(pred)
    st.write(f"A IA acha que você desenhou: {pred_label}")

    # Mostrar probabilidades de cada número
    st.bar_chart(pred.flatten())
