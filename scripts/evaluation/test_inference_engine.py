from src.minigpt.model_loader import load_minigpt
from src.minigpt.inference_engine import InferenceEngine


model, tokenizer, checkpoint, device = load_minigpt(
    "models/minigpt_large_v2_sft_best.pt",
    "models/tokenizer_large.json",
    device="cpu",
)

engine = InferenceEngine(model, tokenizer, device)
engine.start()

import threading


def make_request(i):
    response = engine.submit(
        f"Explica inteligência artificial em poucas palavras. Pedido {i}",
        max_new_tokens=10,
    )
    print(f"\nREQUEST {i}: {response}")


threads = []

for i in range(4):
    t = threading.Thread(target=make_request, args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()