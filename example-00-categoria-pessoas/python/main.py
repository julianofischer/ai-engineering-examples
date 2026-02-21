import numpy as np
import tensorflow as tf
from tensorflow import keras


def train_model(input_xs, output_ys):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_xs.shape[1],)),
        # primeira camada
        # 80 neuronios porque tem poucas amostras para treinamento
        # activation: 'relu' é a função de ativação para a primeira camada
        keras.layers.Dense(80, activation='relu'),
        # saída
        keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(input_xs, output_ys, epochs=100, verbose=0, shuffle=True)

    return model


def predict(model, pessoa_tensor):
    resultado = model.predict(pessoa_tensor, verbose=0)
    return [{'prob': prob, 'index': index} for index, prob in enumerate(resultado[0])]


# Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
# pessoas = [
#     {"nome": "Erick", "idade": 30, "cor": "azul", "localizacao": "São Paulo"},
#     {"nome": "Ana", "idade": 25, "cor": "vermelho", "localizacao": "Rio"},
#     {"nome": "Carlos", "idade": 40, "cor": "verde", "localizacao": "Curitiba"}
# ]

# Vetores de entrada com valores já normalizados e one-hot encoded
# Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
# tensor_pessoas = [
#     [0.33, 1, 0, 0, 1, 0, 0],  # Erick
#     [0, 0, 1, 0, 0, 1, 0],     # Ana
#     [1, 0, 0, 1, 0, 0, 1]      # Carlos
# ]

# Usamos apenas os dados numéricos, como a rede neural só entende números.
# tensor_pessoas_normalizado corresponde ao dataset de entrada do modelo.
tensor_pessoas_normalizado = np.array([
    [0.33, 1, 0, 0, 1, 0, 0],  # Erick
    [0, 0, 1, 0, 0, 1, 0],     # Ana
    [1, 0, 0, 1, 0, 0, 1]      # Carlos
])

# Labels das categorias a serem previstas (one-hot encoded)
# [premium, medium, basic]
labels_nomes = ["premium", "medium", "basic"]
tensor_labels = np.array([
    [1, 0, 0],  # premium - Erick
    [0, 1, 0],  # medium - Ana
    [0, 0, 1]   # basic - Carlos
])

# Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
input_xs = tensor_pessoas_normalizado
output_ys = tensor_labels

print("Input (pessoas normalizadas):")
print(input_xs)
print("\nOutput (labels):")
print(output_ys)

model = train_model(input_xs, output_ys)

pessoa = {"nome": "Zé", "idade": 28, "cor": "verde", "localizacao": "Curitiba"}
# normalizar a idade
# idade_min = 25, idade_max = 40
# formula: (idade - idade_min) / (idade_max - idade_min)
idade_normalizada = (pessoa['idade'] - 25) / (40 - 25)

pessoa_tensor = np.array([[idade_normalizada, 0, 0, 1, 0, 0, 1]])

resultado = predict(model, pessoa_tensor)
resultado_ordenado = sorted(resultado, key=lambda x: x['prob'], reverse=True)
resultado_formatado = [f"{labels_nomes[p['index']]} - {p['prob']:.2f}%" for p in resultado_ordenado]
print("\nPrevisão para Zé:")
print('\n'.join(resultado_formatado))
