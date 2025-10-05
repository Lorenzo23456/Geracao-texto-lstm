#Bibliotecas:
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import re

#----------------------------------------------------------------------------------------
#CONSTANTES
SEQ_LENGTH = 20
VOCAB_SIZE = 5000
BATCH_SIZE = 128
EMBEDDING_DIM = 100 
HIDDEN_DIM = 256    
NUM_LAYERS = 2 
LEARNING_RATE = 0.0005
NUM_EPOCHS = 30


#----------------------------------------------------------------------------------------
#Classes

class TextoDataset(Dataset):
    def __init__(self, entrada, prevista):
        self.entrada = entrada
        self.prevista = prevista
    
    def __len__(self):
        return len(self.entrada)
    
    def __getitem__(self, idx):
        return self.entrada[idx], self.prevista[idx]

class GeradorTextoLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(GeradorTextoLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True, 
            dropout=0.5 if num_layers > 1 else 0 
        )
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward (self, x, hidden=None):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        ultima_saida = lstm_out[:, -1, :] 
        output = self.linear(ultima_saida)
        return output, hidden
#----------------------------------------------------------------------------------------
#Funções

def criagem_corpus ():
    vet = []
    total = 0
    with open("corpus.txt","r",encoding="utf-8") as dados:
        for letra in dados:
            if len(letra) > 1:
                vet.append(letra)
    
    #Info do corpus:
    print(f"Total de músicas: {len(vet)}")
    for i in range(0,len(vet)):
        total += len(vet[i].split())
    print(f"Total de palavras: {total}")
    print(f"Média de palavras por música: {(total/len(vet)):.2f}")

    return vet
def tokenizacao(vetor):
    completo_txt = " ".join(vetor).lower()
    completo_txt = re.sub(r'[^\w\s]|_',' ',completo_txt)
    
    t = re.findall(r'[a-z]{2,}', completo_txt) 

    return t
def mapeamento(size):
    
    #pad = preencher sequências | sos = start of sequence | unk = unknown, ou seja palavras raras.
    v = {"pad":0,"<sos>":1,"<unk>":2} 
    v.update({
        palavra: i + 3
        for i, (palavra,_) in enumerate (size.most_common(VOCAB_SIZE - 3))})
    
    u = {id: palavra for palavra, id in v.items()}

    return v, u
def preparacao_dados(t_id):
    e = []; s = []
    for i in range(0,len(t_id)-SEQ_LENGTH):
        seq_in = tokens_ID[i:i+SEQ_LENGTH]
        seq_out = tokens_ID[i+SEQ_LENGTH]

        e.append(seq_in);s.append(seq_out)
    return e,s
def preparacao_dados_tamanho(e):
    #Treino(80) - Validação(10) - Teste(10)
    # 0.8 = evitar vazamento
    tamanho_treino = int(len(e)*0.8)
    tamanho_val = int(0.1 * len(e)) + tamanho_treino
    return [tamanho_treino,tamanho_val]
def treinar_modelo(mod,d_treino,d_val,crit,opt,n_epocas):
    print("\nIniciando treinamento do Baseline >:D ")

    melhor_loss_val = float("inf")

    for epocas in range(n_epocas):
        mod.train()
        t_loss_treino = 0

        for e_batch, s_batch in d_treino:
            e_batch, s_batch = e_batch.to(device), s_batch.to(device)
            opt.zero_grad()
            out, _ = mod(e_batch)
            loss = crit(out,s_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(mod.parameters(),max_norm= 1.0)
            opt.step()
            t_loss_treino += loss.item() * e_batch.size(0)

        mod.eval()
        t_loss_val = 0

        with torch.no_grad():
            for e_batch_val, s_batch_val in d_val:
                e_batch_val, s_batch_val = e_batch_val.to(device), s_batch_val.to(device)
                output_val, _ = mod(e_batch_val)
                loss_val = crit(output_val,s_batch_val)
                t_loss_val += loss_val.item() * e_batch_val.size(0)

        m_loss_treino = t_loss_treino / len(d_treino.dataset)
        m_loss_val = t_loss_val/len(d_val.dataset)
        print(f"Época {epocas+1}/{n_epocas} \n| Loss Treino: {m_loss_treino:.4f} \n| Loss Val: {m_loss_val:.4f}")

        if m_loss_val < melhor_loss_val:
            melhor_loss_val = m_loss_val
            torch.save(mod.state_dict(), 'modelo_baseline_melhor.pth')

            print(">>> Modelo salvo (Melhor Loss de Validação) <<<")
#----------------------------------------------------------------------------------------
#Script principal 

corpus = []
corpus = criagem_corpus()

tokens = tokenizacao(corpus)
total_palavras = Counter(tokens)

word_to_id , id_to_word = mapeamento(total_palavras) ; # vocab = word_to_id
tokens_ID = [word_to_id.get(token, word_to_id['<unk>']) for token in tokens]

print(f"Tamanho final do vocab: {len(word_to_id)}")
print(f"Total de tokens(ID): {len(tokens_ID)}")

entrada = []; prevista =[]
entrada, prevista = preparacao_dados(tokens_ID)

print(f"\nNúmero total de sequências de treinamento geradas: {len(entrada)}")

#converção para tensores PyTorch
entrada_tensor = torch.LongTensor(entrada)
prevista_tensor = torch.LongTensor(prevista)


tamanho = []
tamanho = preparacao_dados_tamanho(entrada)

#entrada
entrada_treino = entrada_tensor[:tamanho[0]]
entrada_val = entrada_tensor[tamanho[0]:tamanho[1]]
entrada_test = entrada_tensor[tamanho[1]:]

#prevista
prevista_treino = prevista_tensor[:tamanho[0]]
prevista_val = prevista_tensor[tamanho[0]:tamanho[1]]
prevista_test = prevista_tensor[tamanho[1]:]

print(f"Tamanho do Conjunto de Treino: {len(entrada_treino)}")
print(f"Tamanho do Conjunto de Validação: {len(entrada_val)}")
print(f"Tamanho do Conjunto de Teste: {len(entrada_test)}")


#Dataloaders
dataloader_treino = DataLoader(TextoDataset(entrada_treino, prevista_treino), batch_size=BATCH_SIZE, shuffle=True)
dataloader_val = DataLoader(TextoDataset(entrada_val, prevista_val), batch_size=BATCH_SIZE)
dataloader_teste = DataLoader(TextoDataset(entrada_test, prevista_test), batch_size=BATCH_SIZE)

device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

mod_baseline = GeradorTextoLSTM(vocab_size=VOCAB_SIZE,embedding_dim=EMBEDDING_DIM,hidden_dim=HIDDEN_DIM,num_layers=NUM_LAYERS)
mod_baseline.to(device)

criterio = nn.CrossEntropyLoss()
otimizador = torch.optim.Adam(mod_baseline.parameters(), lr=LEARNING_RATE)

treinar_modelo(mod_baseline,dataloader_treino,dataloader_val,criterio,otimizador,NUM_EPOCHS)
