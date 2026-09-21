import torch
import torch.nn as nn
import torch.nn.functional as F

## ATTENTION ##

class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.head_size = head_size
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        scores = q @ k.transpose(-2, -1)
        scores = scores / (self.head_size ** 0.5)

        scores = scores.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        out = weights @ v

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, block_size, dropout):
        super().__init__()

        head_size = n_embd // num_heads

        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size, dropout)
            for _ in range(num_heads)
        ])

        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        outputs = [head(x) for head in self.heads]

        out = torch.cat(outputs, dim=-1)

        out = self.proj(out)

        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, num_heads, block_size, dropout):
        super().__init__()

        self.attention = MultiHeadAttention(
            n_embd,
            num_heads,
            block_size,
            dropout
        )

        self.ffwd = FeedForward(n_embd, dropout)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):

        x = x + self.attention(self.ln1(x))

        x = x + self.ffwd(self.ln2(x))

        return x



class MiniGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, num_heads, block_size, num_layers, dropout):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[
            Block(n_embd, num_heads, block_size, dropout)
            for _ in range(num_layers)
        ])

        self.ln_final = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(n_embd, vocab_size,  bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, idx):
        B, T = idx.shape

        tok_emb = self.token_embedding(idx)       # [B,T,32]

        positions = torch.arange(
            T,
            device=idx.device
        )

        pos_emb = self.position_embedding(positions)  # [T,32]

        x = tok_emb + pos_emb                    # [B,T,32]

        x = self.blocks(x)                       # [B,T,32]

        x = self.ln_final(x)                     # [B,T,32]

        logits = self.lm_head(x)                 # [B,T,45]

        return logits

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):

            # Só podemos usar no máximo block_size tokens
            idx_cond = idx[:, -self.block_size:]

            # Forward
            logits = self(idx_cond)

            # Só queremos a previsão da última posição
            logits = logits[:, -1, :]   # [B, vocab_size]

            # Logits → probabilidades
            probs = F.softmax(logits, dim=-1)

            # Escolher próximo token
            idx_next = torch.multinomial(
                probs,
                num_samples=1
            )

            # Adicionar o novo token ao contexto
            idx = torch.cat((idx, idx_next), dim=1)

        return idx