import torch
import torch.nn as nn
import math

import random

# 固定 random seed
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ==========================================
# 1. 基礎組件：位置編碼 (保持不變)
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

# ==========================================
# 2. Transformer 主架構 (保持不變)
# ==========================================
class SmallTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers, num_decoder_layers=num_layers,
            dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))
        out = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)
        return self.fc_out(out)

# ==========================================
# 3. 資料處理與【解碼】邏輯 (從這裡改)
# ==========================================


# ========== 直接用 string 當訓練資料 ==========
from torch.utils.data import Dataset, DataLoader

def clean_text(text):
    return text.replace('?', '').replace(',', '').replace('.', '')

# 範例訓練資料（每組 context, query, answer）
train_data = [
    (
        "[A] Hi, how are you? [B] I'm good. [A] By the way, my locker code is 7314. [B] Nice. [A] I may change it next month. [B] I always forget mine.",
        "What is Speaker A's locker code?",
        "7314"
    ),
    (
        "[A] Hello! [B] Hi! [A] Do you like pizza? [B] Yes, I love it! [A] Me too.",
        "What does Speaker B like?",
        "pizza"
    ),
    (
        "[A] Good morning. [B] Morning! [A] I just bought a new bike yesterday. [B] That's great! What color is it? [A] It's blue.",
        "What color is Speaker A's bike?",
        "blue"
    ),
    (
        "[A] Hey! [B] Hi! [A] I wake up at 6 AM every day. [B] That's early. [A] Yes, I like mornings.",
        "What time does Speaker A wake up?",
        "6 AM"
    ),
    (
        "[A] Hello. [B] Hi. [A] I have a cat named Luna. [B] Cute name! [A] She is very playful.",
        "What is the name of Speaker A's cat?",
        "Luna"
    ),
    (
        "[A] Hi there. [B] Hello! [A] I work at a hospital. [B] Are you a doctor? [A] Yes, I am.",
        "Where does Speaker A work?",
        "a hospital"
    ),
    (
        "[A] Hey. [B] Hi. [A] My favorite food is sushi. [B] I like it too. [A] It's delicious.",
        "What is Speaker A's favorite food?",
        "sushi"
    ),
    (
        "[A] Hello! [B] Hi! [A] I have two brothers. [B] That's nice. [A] We are close.",
        "How many brothers does Speaker A have?",
        "two"
    ),
    (
        "[A] Hi. [B] Hello. [A] I am reading a book. [B] What book? [A] A mystery novel.",
        "What is Speaker A reading?",
        "a mystery novel"
    ),
    (
        "[A] Hey! [B] Hi! [A] I drive a black car. [B] Cool. [A] I like its color.",
        "What color is Speaker A's car?",
        "black"
    ),
    (
        "[A] Hi. [B] Hello. [A] I live in Canada. [B] That's nice. [A] It's very cold in winter.",
        "Where does Speaker A live?",
        "Canada"
    ),
    (
        "[A] Hello. [B] Hi. [A] I play soccer every weekend. [B] That's fun. [A] I enjoy it.",
        "What sport does Speaker A play?",
        "soccer"
    ),
    (
        "[A] Hi! [B] Hello! [A] I drink tea every morning. [B] Me too. [A] It's relaxing.",
        "What does Speaker A drink every morning?",
        "tea"
    ),
    (
        "[A] Hey. [B] Hi. [A] My birthday is on May 5th. [B] Nice. [A] I love birthdays.",
        "When is Speaker A's birthday?",
        "May 5th"
    ),
    (
        "[A] Hello. [B] Hi. [A] I have a dog. [B] What's its name? [A] Rocky.",
        "What is the name of Speaker A's dog?",
        "Rocky"
    ),
    (
        "[A] Hi. [B] Hello. [A] I go to school by bus. [B] Is it far? [A] Not really.",
        "How does Speaker A go to school?",
        "by bus"
    ),
    (
        "[A] Hey. [B] Hi. [A] I like watching movies. [B] Same here. [A] It's fun.",
        "What does Speaker A like doing?",
        "watching movies"
    ),
    (
        "[A] Hello. [B] Hi. [A] I have three sisters. [B] Wow. [A] Yes, a big family.",
        "How many sisters does Speaker A have?",
        "three"
    ),
    (
        "[A] Hi! [B] Hello! [A] I am cooking dinner. [B] What are you making? [A] Pasta.",
        "What is Speaker A cooking?",
        "pasta"
    ),
    (
        "[A] Hello. [B] Hi. [A] I study English every day. [B] That's good. [A] I want to improve.",
        "What does Speaker A study?",
        "English"
    ),
    (
        "[A] Hi. [B] Hello. [A] I wear glasses. [B] Me too. [A] They help me see.",
        "What does Speaker A wear?",
        "glasses"
    ),
    (
        "[A] Hey. [B] Hi. [A] I have a blue backpack. [B] Nice. [A] I use it daily.",
        "What color is Speaker A's backpack?",
        "blue"
    ),
    (
        "[A] Hello. [B] Hi. [A] I eat breakfast at 8 AM. [B] Same here. [A] It's my routine.",
        "What time does Speaker A eat breakfast?",
        "8 AM"
    ),
    (
        "[A] Hi. [B] Hello. [A] I like music. [B] What kind? [A] Pop music.",
        "What kind of music does Speaker A like?",
        "pop music"
    ),
    (
        "[A] Hello. [B] Hi. [A] I have a laptop. [B] What brand? [A] Apple.",
        "What brand is Speaker A's laptop?",
        "Apple"
    ),
    (
        "[A] Hi. [B] Hello. [A] I am wearing a red shirt. [B] Looks nice. [A] Thank you.",
        "What color is Speaker A's shirt?",
        "red"
    ),
    (
        "[A] Hello. [B] Hi. [A] I sleep at 11 PM. [B] That's late. [A] Yes.",
        "What time does Speaker A sleep?",
        "11 PM"
    ),
    (
        "[A] Hi. [B] Hello. [A] I ride a bicycle. [B] That's healthy. [A] I enjoy it.",
        "What does Speaker A ride?",
        "a bicycle"
    ),
    (
        "[A] Hello. [B] Hi. [A] I have a job. [B] Where? [A] At a bank.",
        "Where does Speaker A work?",
        "a bank"
    ),
    (
        "[A] Hi. [B] Hello. [A] I drink water often. [B] Good habit. [A] Yes.",
        "What does Speaker A drink often?",
        "water"
    ),
    (
        "[A] Hey, what time is it? [B] It's 3 PM. [A] Thanks. [B] No problem.",
        "What time is it?",
        "3 PM"
    ),
    (
        "[A] Do you have a pen? [B] Yes, it's on the table. [A] Thanks. [B] You're welcome.",
        "Where is the pen?",
        "on the table"
    ),
    (
        "[A] What are you eating? [B] I'm eating noodles. [A] Looks tasty. [B] It is.",
        "What is Speaker B eating?",
        "noodles"
    ),
    (
        "[A] Where are you going? [B] I'm going to the library. [A] Study hard. [B] I will.",
        "Where is Speaker B going?",
        "the library"
    ),
    (
        "[A] What's your favorite color? [B] Green. [A] Nice choice. [B] Thanks.",
        "What is Speaker B's favorite color?",
        "green"
    ),
    (
        "[A] How old are you? [B] I'm 20 years old. [A] Cool. [B] Thanks.",
        "How old is Speaker B?",
        "20"
    ),
    (
        "[A] What day is it today? [B] It's Monday. [A] Oh, I have work. [B] Good luck.",
        "What day is it?",
        "Monday"
    ),
    (
        "[A] Do you like coffee or tea? [B] I prefer tea. [A] Me too. [B] It's relaxing.",
        "What does Speaker B prefer?",
        "tea"
    ),
    (
        "[A] Where did you buy that? [B] At the mall. [A] Nice. [B] It was cheap.",
        "Where did Speaker B buy it?",
        "the mall"
    ),
    (
        "[A] What are you watching? [B] A movie. [A] Is it good? [B] Yes.",
        "What is Speaker B watching?",
        "a movie"
    ),
    (
        "[A] How do you go to work? [B] By train. [A] Is it fast? [B] Yes.",
        "How does Speaker B go to work?",
        "by train"
    ),
    (
        "[A] What is your job? [B] I'm a teacher. [A] That's great. [B] I like it.",
        "What is Speaker B's job?",
        "a teacher"
    ),
    (
        "[A] What do you do on weekends? [B] I play basketball. [A] Sounds fun. [B] It is.",
        "What does Speaker B do on weekends?",
        "play basketball"
    ),
    (
        "[A] What is in your bag? [B] Books. [A] Heavy? [B] A little.",
        "What is in Speaker B's bag?",
        "books"
    ),
    (
        "[A] Do you have siblings? [B] Yes, one sister. [A] Nice. [B] She's kind.",
        "How many siblings does Speaker B have?",
        "one"
    ),
    (
        "[A] What time do you sleep? [B] Around 10 PM. [A] That's early. [B] Yes.",
        "What time does Speaker B sleep?",
        "10 PM"
    ),
    (
        "[A] What language do you speak? [B] English and Spanish. [A] Impressive. [B] Thanks.",
        "What languages does Speaker B speak?",
        "English and Spanish"
    ),
    (
        "[A] What is your hobby? [B] Drawing. [A] Cool. [B] I enjoy it.",
        "What is Speaker B's hobby?",
        "drawing"
    ),
    (
        "[A] Where is your phone? [B] In my pocket. [A] Okay. [B] Yep.",
        "Where is Speaker B's phone?",
        "in my pocket"
    ),
    (
        "[A] What did you drink? [B] Orange juice. [A] Healthy. [B] Yes.",
        "What did Speaker B drink?",
        "orange juice"
    ),
]



class QADataset(Dataset):
    def __init__(self, data, vocab=None, max_len=64):
        self.samples = data
        # 建立詞表
        if vocab is None:
            all_text = []
            for c, q, a in self.samples:
                all_text.extend(clean_text(c).split())
                all_text.extend(clean_text(q).split())
                all_text.extend(clean_text(a).split())
            unique_words = sorted(list(set(all_text)))
            vocab = {word: i+4 for i, word in enumerate(unique_words)}
            vocab["<PAD>"], vocab["<SOS>"], vocab["<EOS>"], vocab["<UNK>"] = 0, 1, 2, 3
        self.vocab = vocab
        self.idx2word = {idx: word for word, idx in vocab.items()}
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def tokenize(self, text):
        ids = [self.vocab.get(w, self.vocab["<UNK>"]) for w in clean_text(text).split()]
        return ids[:self.max_len]

    def __getitem__(self, idx):
        context, query, answer = self.samples[idx]
        src = self.tokenize(context + " [SEP] " + query)
        tgt = [self.vocab["<SOS>"]] + self.tokenize(answer) + [self.vocab["<EOS>"]]
        src += [self.vocab["<PAD>"]] * (self.max_len - len(src))
        tgt += [self.vocab["<PAD>"]] * (self.max_len - len(tgt))
        return torch.tensor(src), torch.tensor(tgt)

# 準備資料集
dataset = QADataset(train_data)
print("建議：請確保所有答案格式一致（如全部小寫、無多餘空白），否則模型難以學習！")
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
vocab = dataset.vocab
idx2word = dataset.idx2word
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmallTransformer(vocab_size=len(vocab), num_layers=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"])
# 簡單訓練 200 epoch
for epoch in range(200):
    model.train()
    total_loss = 0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        # decoder input: 去掉最後一個 token
        output = model(src, tgt[:, :-1])
        # output: (batch, seq, vocab) -> (batch*seq, vocab)
        output = output.reshape(-1, output.size(-1))
        # target: 去掉第一個 token
        target = tgt[:, 1:].reshape(-1)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(dataloader):.4f}")
# 儲存模型
torch.save(model.state_dict(), "small_transformer.pt")
print("訓練完成，模型已儲存 small_transformer.pt")

# ========== 改善後的推論流程：用訓練時的詞表與模型參數 ==========
print("\n===== 測試模型記憶力（用訓練時的詞表與參數） =====")
model = SmallTransformer(vocab_size=len(vocab), num_layers=2).to(device)
model.load_state_dict(torch.load("small_transformer.pt", map_location=device))
model.eval()

def detokenize(ids, idx2word):
    # 將 token id 轉回單字，遇到 <EOS> 就停
    words = []
    for i in ids:
        w = idx2word.get(i, "<UNK>")
        if w == "<EOS>":
            break
        if w not in ["<SOS>", "<PAD>"]:
            words.append(w)
    return " ".join(words)


# 統計正確率
correct = 0
total = len(train_data)
for i, (context, query, answer) in enumerate(train_data):
    src = dataset.tokenize(context + " [SEP] " + query)
    src_tensor = torch.tensor([src]).to(device)
    generated = [vocab["<SOS>"]]
    max_gen_len = 16
    for _ in range(max_gen_len):
        tgt_tensor = torch.tensor([generated]).to(device)
        with torch.no_grad():
            output = model(src_tensor, tgt_tensor)
            next_token_logits = output[0, -1, :]
            next_id = next_token_logits.argmax().item()
        generated.append(next_id)
        if next_id == vocab["<EOS>"]:
            break
    pred_answer = detokenize(generated, idx2word)
    print(f"\n--- 測試樣本 {i+1} ---")
    print(f"Query: {query}")
    print(f"正確 Answer: {answer}")
    print(f"模型預測: {pred_answer}")
    # 簡單比對（忽略大小寫與前後空白）
    if pred_answer.strip().lower() == answer.strip().lower():
        correct += 1
print(f"\n測試正確率: {correct}/{total} = {correct/total:.2%}")
