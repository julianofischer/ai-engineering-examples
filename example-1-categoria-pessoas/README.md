# Exemplo 1: Categorização de Pessoas

Este exemplo demonstra como usar uma rede neural simples para classificar pessoas em categorias (premium, medium, basic) com base em suas características.

## Conceito

O modelo aprende a associar características de uma pessoa (idade, cor favorita, localização) a uma categoria de cliente. É um exemplo didático de classificação multiclasse usando redes neurais.

## Estrutura dos Dados

### Entrada (Features)

Cada pessoa é representada por um vetor de 7 valores:

| Posição | Descrição |
|---------|-----------|
| 0 | Idade normalizada (0-1) |
| 1 | Cor azul (one-hot) |
| 2 | Cor vermelho (one-hot) |
| 3 | Cor verde (one-hot) |
| 4 | Localização São Paulo (one-hot) |
| 5 | Localização Rio (one-hot) |
| 6 | Localização Curitiba (one-hot) |

**Normalização da idade:**
```
idade_normalizada = (idade - idade_min) / (idade_max - idade_min)
```
Onde `idade_min = 25` e `idade_max = 40`.

### Saída (Labels)

Categorias em formato one-hot:
- `[1, 0, 0]` = Premium
- `[0, 1, 0]` = Medium
- `[0, 0, 1]` = Basic

## Dados de Treino

| Nome   | Idade | Cor      | Localização | Categoria |
|--------|-------|----------|-------------|-----------|
| Erick  | 30    | Azul     | São Paulo   | Premium   |
| Ana    | 25    | Vermelho | Rio         | Medium    |
| Carlos | 40    | Verde    | Curitiba    | Basic     |

## Arquitetura do Modelo

```
Input (7 features)
    ↓
Dense (80 neurônios, ReLU)
    ↓
Dense (3 neurônios, Softmax)
    ↓
Output (3 categorias)
```

- **Camada oculta:** 80 neurônios com ativação ReLU
- **Camada de saída:** 3 neurônios com ativação Softmax (uma para cada categoria)
- **Otimizador:** Adam
- **Loss:** Categorical Crossentropy
- **Épocas:** 100

## Como Executar

### JavaScript

```bash
cd js
npm install
node index.js
```

### Python

```bash
cd python
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
python main.py
```


## Exemplo de Previsão

Para uma nova pessoa "Zé" com:
- Idade: 28 (normalizado: 0.2)
- Cor: Verde
- Localização: Curitiba

O modelo retorna a probabilidade de cada categoria, ordenada da mais provável para a menos provável.
