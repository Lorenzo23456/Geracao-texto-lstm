Claro! Aqui está uma versão mais bonita e organizada do seu relatório em Markdown, com melhor estrutura visual, uso de emojis para destacar seções e tabelas para facilitar a leitura:

---

# 🧠 Geração de Textos com Redes Neurais Recorrentes  
## Relatório de Implementação e Avaliação de Modelo LSTM

### 👥 Autores e Afiliação

- **Autores**: Arthur Lima de Menezes, João Pedro Huppes Arenales, Lorenzo de Castro, Pâmela da Silva Paes  
- **Afiliação**: Faculdade de Computação — Universidade Federal de Mato Grosso do Sul (UFMS)

---

### 📌 Resumo do Projeto

Este projeto abordou a implementação e avaliação de um modelo de linguagem baseado em LSTM (Long Short-Term Memory) para geração de texto em português. O processo envolveu desde a construção do corpus até a análise crítica dos resultados.

- **Corpus**: 441 letras de Zeca Pagodinho  
- **Embeddings**: Aprendidos do zero  
- **Motivação**: Vocabulário rico e estilo cultural marcante do artista

---

### 📊 Resultados Principais

- **🔺 Alta Perplexidade**: PPL ≈ 654.4384 no conjunto de teste  
- **⚠️ Overfitting**: Perda no treino caiu, enquanto a de validação aumentou  
- **📝 Qualidade dos Textos**: Baixa coerência e alta repetição

---

### 🧪 Materiais e Métodos

#### 🎼 Corpus: Letras de Zeca Pagodinho

- **Coleta**: Web scraping com BeautifulSoup  
- **Total**: 441 músicas → 81.348 palavras após limpeza

#### 🔧 Pré-processamento

- Tokenização por palavras  
- Vocabulário limitado às 5.000 palavras mais frequentes  
- Palavras raras mapeadas para `<unk>`

#### 🏗️ Arquitetura do Modelo LSTM

| Componente         | Descrição                                                                 |
|--------------------|---------------------------------------------------------------------------|
| Embedding          | Vetores densos de 100 dimensões, aprendidos do zero                       |
| LSTM               | 2 camadas com 256 unidades ocultas cada                                   |
| Camada Linear      | Mapeia saída da LSTM para distribuição de probabilidade do vocabulário    |

#### ⚙️ Treinamento

- **Épocas**: 15  
- **Otimizador**: Adam (lr = 0.0005)  
- **Função de Perda**: Cross-Entropy Loss

---

### 📈 Resultados e Discussão

#### 🔢 Performance Quantitativa

- **Perplexidade (PPL)**: 654.4384  
- **Interpretação**: Valor alto → dificuldade em prever sequências não vistas → overfitting

#### 🧾 Performance Qualitativa: Geração de Texto

| Temperatura | Característica Principal                          | Coerência                      | Repetição                          |
|-------------|----------------------------------------------------|--------------------------------|------------------------------------|
| 0.7         | Prioriza palavras mais prováveis                   | Baixa (coerência de longo prazo) | Alta ("eu não vou não vou...")     |
| 1.0         | Equilíbrio entre coerência e criatividade          | Fraca (transições abruptas)     | Média                              |
| 1.5         | Mais aleatoriedade, suaviza probabilidades         | Muito baixa (saltos temáticos)  | Baixa (alta variedade de palavras) |

- Vocabulário temático do samba foi mantido

---

### 🚧 Limitações do Modelo

- **Corpus Pequeno**: 441 músicas não foram suficientes para generalização  
- **Repetição e Incoerência**: Erros predominantes  
- **Viés Temático**: Palavras como "samba", "amor", "cerveja" dominam  
- **Vocabulário Fixo**: Uso de `<unk>` limitou riqueza lexical

---

### ✅ Conclusão

O modelo LSTM conseguiu capturar o estilo e vocabulário temático de Zeca Pagodinho, mas apresentou:

- **Alta perplexidade**
- **Baixa qualidade textual**
- **Overfitting severo**

🔍 A principal lição é que **modelos de linguagem treinados do zero exigem grandes volumes de dados** para alcançar boa generalização.

---

Se quiser, posso transformar esse conteúdo em uma apresentação ou pôster acadêmico também!
