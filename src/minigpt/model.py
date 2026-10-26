import torch
import torch.nn as nn
import torch.nn.functional as F

## ATTENTION ##

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, block_size, dropout):
        super().__init__()

        assert n_embd % num_heads == 0

        self.num_heads = num_heads
        self.head_size = n_embd // num_heads
        self.dropout = dropout

        # Calculamos Q, K e V de todos os heads de uma vez
        self.qkv = nn.Linear(
            n_embd,
            3 * n_embd,
            bias=False
        )

        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x, past_kv=None, use_cache=False, attention_mask=None, ):
        B, T, C = x.shape

        # [B, T, C] -> [B, T, 3C]
        qkv = self.qkv(x)

        # -> três tensores [B, T, C]
        q, k, v = qkv.chunk(3, dim=-1)

        # [B,T,C] -> [B,T,H,D] -> [B,H,T,D]
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv

            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv = (k, v) if use_cache else None


        if attention_mask is not None:

            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )

        else:

            is_causal = past_kv is None

            out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=is_causal,
            )

        # [B,H,T,D] -> [B,T,H,D] -> [B,T,C]
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.proj(out)

        if use_cache:
            return out, new_kv

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

    def forward(
        self,
        x,
        past_kv=None,
        use_cache=False,
        attention_mask=None,
    ):

        if use_cache:
            attention_out, new_kv = self.attention(
            self.ln1(x),
            past_kv=past_kv,
            use_cache=True,
            attention_mask=attention_mask,
        )
        else:
            attention_out = self.attention(
                self.ln1(x)
            )
            new_kv = None

        x = x + attention_out
        x = x + self.ffwd(self.ln2(x))

        if use_cache:
            return x, new_kv

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

    def forward(
        self,
        idx,
        past_key_values=None,
        use_cache=False,
        attention_mask=None,
        position_ids=None,
    ):
        B, T = idx.shape

        tok_emb = self.token_embedding(idx)       # [B,T,32]

        if position_ids is not None:

            positions = position_ids

        else:

            past_length = 0

            if past_key_values is not None:
                past_length = past_key_values[0][0].shape[2]

            positions = torch.arange(
                past_length,
                past_length + T,
                device=idx.device,
            )

        pos_emb = self.position_embedding(positions)  # [T,32]

        x = tok_emb + pos_emb                    # [B,T,32]

        new_past_key_values = []

        for i, block in enumerate(self.blocks):

            past_kv = None

            if past_key_values is not None:
                past_kv = past_key_values[i]

            if use_cache:
                x, new_kv = block(
                    x,
                    past_kv=past_kv,
                    use_cache=True,
                    attention_mask=attention_mask,

                )

                new_past_key_values.append(new_kv)

            else:
                x = block(x)

        x = self.ln_final(x)                     # [B,T,32]

        logits = self.lm_head(x)                 # [B,T,45]

        if use_cache:
            return logits, new_past_key_values

        return logits

    def generate(self, idx, max_new_tokens, temperature=1.0, eos_id=None, ):

        # PREFILL: processamos o prompt inteiro UMA vez
        logits, past_key_values = self(
            idx,
            use_cache=True,
        )


        for _ in range(max_new_tokens):

            # Escolher próximo token
            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)

            idx_next = torch.multinomial(
                probs,
                num_samples=1,
            )

            if eos_id is not None and idx_next.item() == eos_id:
                break

            # Guardamos token no output final
            idx = torch.cat(
                (idx, idx_next),
                dim=1,
            )

            # DECODE:
            # só o token novo entra no modelo
            logits, past_key_values = self(
                idx_next,
                past_key_values=past_key_values,
                use_cache=True,
            )

        return idx