import torch
import torch.nn as nn
import math

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

# 1. 定義數據
turns = [
    "[A] Hi, how are you?",
    "[B] I'm good.",
    "[A] By the way, my locker code is 7314.",
    "[B] Nice.",
    "[A] I may change it next month.",
    "[B] I always forget mine.",
]
context = " ".join(turns)
query = "What is Speaker A's locker code?"

# 2. 建立 Vocab 與 反向字典 (idx2word)
# 先簡單清洗標點，確保 7314 能被正確切出來
def clean_text(text):
    return text.replace('?', '').replace(',', '').replace('.', '')

full_text = clean_text(context + " " + query + " [SEP]")
unique_words = sorted(list(set(full_text.split())))

# 建立 Word -> ID
vocab = {word: i+4 for i, word in enumerate(unique_words)}
vocab["<PAD>"], vocab["<SOS>"], vocab["<EOS>"], vocab["<UNK>"] = 0, 1, 2, 3

# --- 這裡是你缺少的關鍵：ID -> Word ---
idx2word = {idx: word for word, idx in vocab.items()}

def tokenize(text):
    return [vocab.get(w, vocab["<UNK>"]) for w in clean_text(text).split()]

# 3. 轉換為 Tensor
src_ids = tokenize(context + " [SEP] " + query)
src_tensor = torch.tensor([src_ids]) 
tgt_tensor = torch.tensor([[vocab["<SOS>"]]]) # 起始符號

# 4. 初始化模型與執行
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmallTransformer(vocab_size=len(vocab), num_layers=2).to(device)

model.eval()
with torch.no_grad():
    output = model(src_tensor.to(device), tgt_tensor.to(device))
    
    # 5. 解碼預測結果
    # output shape: (batch, tgt_seq_len, vocab_size)
    # 取最後一個時間點的預測結果
    last_token_logits = output[0, -1, :] 
    pred_id = last_token_logits.argmax().item()
    
    # 將 ID 轉回 單字
    predicted_word = idx2word.get(pred_id, "<UNK>")

print(f"輸入序列長度: {src_tensor.shape[1]}")
print(f"預測的 Token ID: {pred_id}")
print(f"對應的單字結果: 【 {predicted_word} 】")