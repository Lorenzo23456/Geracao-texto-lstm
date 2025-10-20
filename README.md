# Geração de Textos com Redes Neurais Recorrentes: Relatório de Implementação e Avaliação de Modelo LSTM

### Autores e Afiliação

    Autores: Arthur Lima de Menezes, João Pedro Huppes Arenales, Lorenzo de Castro, Pâmela da Silva Paes 

### Afiliação: Faculdade de Computação - Universidade Federal de Mato Grosso do Sul (UFMS)

 ### Resumo do Projeto

Este projeto detalhou a implementação e avaliação de um modelo de linguagem LSTM (Long Short-Term Memory) para a tarefa de geração de texto em português. O objetivo foi desenvolver o processo completo, desde a construção do corpus até a análise crítica do modelo.

O modelo foi treinado com um corpus customizado de 441 letras de música de Zeca Pagodinho, utilizando embeddings aprendidas do zero. A escolha do artista se deu pela riqueza de seu vocabulário e estilo cultural marcante.

Resultados Principais:

    Alta Perplexidade: O modelo obteve uma perplexidade (PPL) de aproximadamente 654.4384 no conjunto de teste.

Overfitting: A análise das curvas de perda e a alta perplexidade indicaram forte overfitting. A perda no treino diminuiu, enquanto a perda na validação aumentou após as primeiras épocas.

Qualidade do Texto: Os textos gerados apresentaram baixa coerência e alta repetição.

Materiais e Métodos

Corpus: Letras de Zeca Pagodinho

    Coleta: Foram extraídas 441 letras de música via web scraping usando a biblioteca BeautifulSoup.

Conteúdo: Após limpeza e normalização, o corpus foi consolidado em um arquivo de texto e totalizou 81.348 palavras.

Pré-processamento e Tokenização

    O texto foi dividido em tokens (palavras).

Foi criado um vocabulário de 5.000 palavras mais frequentes; as demais foram mapeadas para o token de desconhecido (<unk>).

Arquitetura do Modelo Base LSTM

O modelo implementado é uma rede neural LSTM, projetada para prever a próxima palavra da sequência. A arquitetura consistiu em:

    Camada de Embedding: Transforma índices numéricos em vetores densos de 100 dimensões. Os pesos foram aprendidos do zero durante o treinamento.

Camada LSTM: O núcleo do modelo, com duas camadas de células LSTM, cada uma com 256 unidades ocultas.

Camada Linear: Mapeia a saída da LSTM para o tamanho do vocabulário, gerando a distribuição de probabilidade da próxima palavra.

Treinamento

    Épocas: O modelo foi treinado por 15 épocas.

Otimizador: Adam, com taxa de aprendizado inicial de 0.0005.

Função de Perda: Cross-Entropy Loss, métrica padrão para classificação multiclasse (previsão da próxima palavra).

### Resultado e Discussão

Performance Quantitativa: Perplexidade (PPL)

    PPL Obtida: 654.4384.

Interpretação: Este valor é considerado alto, indicando que o modelo tem dificuldade em prever sequências de palavras que não viu no treino. A principal causa é o overfitting, que é comum ao treinar um modelo do zero em um corpus relativamente pequeno (441 músicas).

Performance Qualitativa: Geração de Amostras

A geração de textos com diferentes temperaturas revelou:
Temperatura	Característica Principal	Coerência	Repetição
0.7	

Prioriza palavras mais prováveis.

	

Baixa (Coerência de Longo Prazo).
	

		

Alta ("eu não vou não vou não vou...").

1.0	

Equilíbrio entre coerência e criatividade.
	

	

Fraca (Transições abruptas entre conjuntos temáticos).
	

		

Média (Menos repetitivo que 0.7).

1.5	

Suaviza probabilidades, aumentando a aleatoriedade.
	

		

Mais Baixa (Priorização da aleatoriedade leva a saltos temáticos).
	

	

Baixa (Alta variedade de palavras).

O modelo foi capaz de manter o vocabulário temático do samba.

Limitações do Modelo Base

    Tamanho do Corpus: As 441 músicas foram insuficientes para que o modelo aprendesse as complexas regras gramaticais e semânticas da língua portuguesa, resultando no overfitting.

Repetição e Incoerência: O principal erro foi a repetição de palavras e estruturas frasais, além da incoerência semântica predominante.

Viés Temático: O modelo demonstrou forte viés temático, refletindo o conteúdo das letras de Zeca Pagodinho (palavras como "samba", "amor", "cerveja", "deixa" e "vida" apareceram com altíssima frequência).

Vocabulário Fixo: Mapear termos fora do conjunto das 5.000 palavras mais frequentes para o token <unk> limitou a capacidade da rede de gerar um vocabulário mais rico.

### Conclusão

O estudo demonstrou que o modelo LSTM treinado do zero conseguiu capturar o estilo e vocabulário temático do corpus de Zeca Pagodinho. Contudo, a alta perplexidade (654.4384) e a baixa qualidade dos textos gerados confirmaram o overfitting, que é uma consequência direta do tamanho pequeno do corpus de treinamento.

A principal conclusão é que treinar um modelo de linguagem a partir do zero em um corpus de nicho e pequena escala é uma tarefa difícil, sublinhando a importância da quantidade massiva de dados para a generalização eficaz.
