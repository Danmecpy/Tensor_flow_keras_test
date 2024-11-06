# Importando bibliotecas necessárias
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

# 1. Definir o Problema e os Objetivos
# Objetivo: Classificar imagens de dígitos de 0 a 9

# 2. Coletar e Preparar os Dados
# Carregar o conjunto de dados MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Pré-processamento: Normalizando e categorizar os rótulos
train_images = train_images / 255.0
test_images = test_images / 255.0
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# 3. Escolher e Preparar Algoritmos
# Definir o modelo com redes neurais
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compilar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinamento do Modelo
history = model.fit(train_images, train_labels, epochs=10, validation_split=0.2)

# 4. Avaliação do Modelo
# Avaliar o modelo com métricas de desempenho
test_loss, test_acc = model.evaluate(test_images, test_labels)
predictions = model.predict(test_images)
print("Acurácia do modelo:", test_acc)
print(classification_report(test_labels.argmax(axis=1), predictions.argmax(axis=1)))

# 5. Implementação e Monitoramento
# Neste caso, salvar o modelo para produção
model.save("mnist_model_tensorflow.h5")
