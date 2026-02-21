import tf from '@tensorflow/tfjs-node';

async function trainModel(inputXs, outputYs) {
    const model = tf.sequential()
    // primeira camada
    // 80 neuronios porque tem poucas amostras para treinamento
    // activation: 'relu' é a função de ativação para a primeira camada
    model.add(tf.layers.dense({ units: 80, inputShape: [inputXs.shape[1]], activation: 'relu' }))

    // saída
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }))

    model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] })

    await model.fit(inputXs, outputYs, { epochs: 100, verbose: 0, shuffle: true })

    return model
}

async function predict(model, pessoaTensor) {
    const resultado = await model.predict(pessoaTensor)
    const predArray = await resultado.array()
    return predArray[0].map((prob, index) => ({ prob, index }))
}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

inputXs.print()
outputYs.print()

const model = await trainModel(inputXs, outputYs)

const pessoa = { nome: "Zé", idade: 28, cor: "verde", localizacao: "Curitiba" }
// normalizar a idade
// idade_min = 25, idade_max = 40

const pessoaTensor = tf.tensor2d([[0.2, 0, 0, 1, 0, 0, 1]])

const resultado = await predict(model, pessoaTensor)
const resultadoOrdenado = resultado.sort((a, b) => b.prob - a.prob).map(p => `${labelsNomes[p.index]} - ${p.prob.toFixed(2)}%`)
console.log(resultadoOrdenado.join('\n'))
