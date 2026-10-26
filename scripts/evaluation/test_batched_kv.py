import torch

from src.minigpt.model_loader import load_minigpt

MODEL_PATH = "models/minigpt_v1_inference.pt"
TOKENIZER_PATH = "models/tokenizer_large.json"

device = torch.device("cpu")

model, tokenizer, checkpoint, device = load_minigpt(
    MODEL_PATH,
    TOKENIZER_PATH,
    device="cpu",
)

model.eval()


prompts = [
    "Olá",
    "O que é inteligência artificial?",
    "Explica-me de forma simples como funciona uma rede neuronal.",
]


# --------------------------------------------------
# PREFILL INDIVIDUAL
# --------------------------------------------------

caches = []
next_tokens = []
cache_lengths = []

with torch.inference_mode():

    for prompt in prompts:

        ids = tokenizer.encode(prompt).ids

        x = torch.tensor(
            [ids],
            dtype=torch.long,
            device=device,
        )

        logits, cache = model(
            x,
            use_cache=True,
        )

        # Greedy para o teste ser determinístico
        token = torch.argmax(
            logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        caches.append(cache)
        next_tokens.append(token)

        cache_lengths.append(
            cache[0][0].shape[2]
        )


# --------------------------------------------------
# REFERÊNCIA:
# decode individual de cada sequência
# --------------------------------------------------

individual_logits = []

with torch.inference_mode():

    for token, cache in zip(next_tokens, caches):

        logits, _ = model(
            token,
            past_key_values=cache,
            use_cache=True,
        )

        individual_logits.append(
            logits[:, -1, :]
        )


# --------------------------------------------------
# BATCHED KV CACHE
# --------------------------------------------------

max_len = max(cache_lengths)
num_layers = len(caches[0])

batched_cache = []

for layer in range(num_layers):

    keys = []
    values = []

    for cache, length in zip(
        caches,
        cache_lengths,
    ):

        k, v = cache[layer]

        pad = max_len - length

        if pad > 0:

            k = torch.nn.functional.pad(
                k,
                (0, 0, pad, 0),
            )

            v = torch.nn.functional.pad(
                v,
                (0, 0, pad, 0),
            )

        keys.append(k)
        values.append(v)

    batched_cache.append(
        (
            torch.cat(keys, dim=0),
            torch.cat(values, dim=0),
        )
    )


tokens = torch.cat(
    next_tokens,
    dim=0,
)


position_ids = torch.tensor(
    cache_lengths,
    dtype=torch.long,
    device=device,
).unsqueeze(1)


mask = torch.zeros(
    len(prompts),
    1,
    1,
    max_len + 1,
    dtype=torch.bool,
    device=device,
)


for i, length in enumerate(cache_lengths):

    pad = max_len - length

    mask[
        i,
        :,
        :,
        pad:max_len
    ] = True

    mask[
        i,
        :,
        :,
        max_len
    ] = True


with torch.inference_mode():

    batched_logits, _ = model(
        tokens,
        past_key_values=batched_cache,
        use_cache=True,
        attention_mask=mask,
        position_ids=position_ids,
    )


# --------------------------------------------------
# COMPARAÇÃO
# --------------------------------------------------

print("\n--- BATCHED KV CORRECTNESS ---")
print("Cache lengths:", cache_lengths)

for i in range(len(prompts)):

    diff = (
        individual_logits[i]
        - batched_logits[i:i+1, -1, :]
    ).abs().max().item()

    print(
        f"Prompt {i + 1}: "
        f"max difference = {diff:.8f}"
    )