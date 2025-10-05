# Bibliotecas
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from collections import Counter
import re
import os 
#----------------------------------------------------
#Constantes 
MODELO = 'modelo_baseline_melhor.pth'

SEQ_LENGTH = 20
VOCAB_SIZE = 5000
EMBEDDING_DIM = 100 
HIDDEN_DIM = 256    
NUM_LAYERS = 2 

#----------------------------------------------------
#Classes

class TextoDataset(Dataset):
    def __init__(self, entrada, prevista):
        self.entrada = entrada
        self.prevista = prevista
    def __len__(self):
        return len(self.entrada)
    def __getitem__(self,idx):
        return self.entrada[idx],self.prevista[idx]
class GeradorTextoLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim,hidden_size=hidden_dim,num_layers=num_layers,batch_first=True,dropout=0.5 if num_layers > 1 else 0 )
        self.linear = nn.Linear(hidden_dim, vocab_size)
    def forward (self, x, hidden=None):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        ultima_saida = lstm_out[:, -1, :] 
        output = self.linear(ultima_saida)
        return output, hidden

#----------------------------------------------------
#Funções
def criagem_corpus ():
    vet = []
    total = 0
    with open("corpus.txt","r",encoding="utf-8") as dados:
        for letra in dados:
            if len(letra) > 1:
                vet.append(letra)
    return vet
def tokenizacao(vetor):
    completo_txt = " ".join(vetor).lower()
    completo_txt = re.sub(r'[^\w\s]|_',' ',completo_txt)
    
    t = re.findall(r'[a-z]{2,}', completo_txt) 

    return t
def mapeamento(size):
    #pad = preencher sequências | sos = start of sequence | unk = unknown, ou seja palavras raras.
    v = {"<pad>":0,"<sos>":1,"<unk>":2} 
    v.update({
        palavra: i + 3
        for i, (palavra,_) in enumerate (size.most_common(VOCAB_SIZE - 3))})
    
    u = {id: palavra for palavra, id in v.items()}

    return v, u
def gerar_texto(m,start_txt,num_word_gen,temp,wti,itw,seq_length):
    m.eval()
    tokens_start = re.findall(r'\b\w+\b', start_txt.lower())
    
    if len(tokens_start) < seq_length:
        input_tokens = [wti["<pad>"]] * (seq_length - len(tokens_start)) + [wti.get(t, wti["<unk>"]) for t in tokens_start]
    else:
        input_tokens = [wti.get(t, wti["<unk>"]) for t in tokens_start[-seq_length:]]
    
    input_tensor = torch.LongTensor(input_tokens).unsqueeze(0).to(device)
    txt_gerado = list(tokens_start)
    hidden = None
    with torch.no_grad():
        for _ in range(num_word_gen):
            output, hidden = m(input_tensor,hidden)

            output_dist = output.squeeze().div(temp).exp()
            next_word_id = torch.multinomial(output_dist,1).item()

            input_tokens.pop(0)
            input_tokens.append(next_word_id)
            input_tensor = torch.LongTensor(input_tokens).unsqueeze(0).to(device)

            txt_gerado.append(itw.get(next_word_id,"<unk>"))
    
    txt_gerado = " ".join(txt_gerado)
    return txt_gerado
#----------------------------------------------------
#Script principal 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")

if not os.path.exists(MODELO):
    print(f"ERRO: Arquivo do Modelo '{MODELO}' não encontrado.")
else:
    corpus = criagem_corpus()
    tokens = tokenizacao(corpus)
    total_palavras = Counter(tokens)
    word_to_id, id_to_word = mapeamento(total_palavras)

    mod = GeradorTextoLSTM(vocab_size= VOCAB_SIZE,embedding_dim= EMBEDDING_DIM,hidden_dim= HIDDEN_DIM,num_layers= NUM_LAYERS)
    mod.to(device)
    mod.load_state_dict(torch.load(MODELO, map_location=device))
    print(f"Modelo treinado '{MODELO}' carregado com sucesso.")

    start_seed = "quando a noite cai e a lua"
    num_to_gerador = 80
    print("\n--- INICIANDO GERAÇÃO (Avaliação Qualitativa) ---")



    temp_7 = gerar_texto(mod, start_seed, num_to_gerador, 0.7, word_to_id, id_to_word, SEQ_LENGTH)
    print(f"\n[T=0.7] (Foco em Coerência):\n{temp_7}")

    temp_10 = gerar_texto(mod, start_seed, num_to_gerador, 1.0, word_to_id, id_to_word, SEQ_LENGTH)
    print(f"\n[T=1.0] (Padrão):\n{temp_10}")

    temp_13 = gerar_texto(mod, start_seed, num_to_gerador, 1.3, word_to_id, id_to_word, SEQ_LENGTH)
    print(f"\n[T=1.3] (Foco em Criatividade):\n{temp_13}")
