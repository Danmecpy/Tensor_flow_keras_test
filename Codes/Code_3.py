import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

# Carregar o conjunto de dados MNIST
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Mostrando as primeiras imagens e rótulos com o Pandas
train_df = pd.DataFrame({
    'Label': train_labels[:10],
    'Image Data': [train_images[i] for i in range(10)]
})
print("Exemplo de dados de treinamento:")
print(train_df)

# Normalização dos dados
train_images = train_images / 255.0
test_images = test_images / 255.0

# Converter os rótulos em categorias (one-hot encoding)
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# Mostrando os rótulos após one-hot encoding
labels_df = pd.DataFrame(train_labels[:10], columns=[f"Class {i}" for i in range(10)])
print("\nRótulos após one-hot encoding:")
print(labels_df)

# Definindo o modelo
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compilar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo e mostrar o progresso
history = model.fit(train_images, train_labels, epochs=3, validation_split=0.1)

# Avaliar o modelo nos dados de teste
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("\nTest accuracy:", test_acc)

# Previsões do modelo
predictions = model.predict(test_images)
predictions_df = pd.DataFrame(predictions[:10], columns=[f"Class {i}" for i in range(10)])
print("\nExemplo de previsões:")
print(predictions_df)

# Exibir relatório de classificação
predicted_classes = predictions.argmax(axis=1)
true_classes = test_labels.argmax(axis=1)
report = classification_report(true_classes, predicted_classes, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print("\nRelatório de Classificação:")
print(report_df)
