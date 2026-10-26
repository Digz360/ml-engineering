import time
import requests

BASE_URL = "http://localhost:8000"

PROMPTS = [
    "O que e machine learning?",
    "Explica redes neuronais.",
    "O que e um transformer?",
]

MAX_NEW_TOKENS = 20


# -------------------------
# Individual requests
# -------------------------

start = time.perf_counter()

for prompt in PROMPTS:
    response = requests.post(
        f"{BASE_URL}/generate",
        json={
            "prompt": prompt,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
        timeout=120,
    )

    response.raise_for_status()

individual_time = time.perf_counter() - start


# -------------------------
# Batch request
# -------------------------

start = time.perf_counter()

response = requests.post(
    f"{BASE_URL}/generate-batch",
    json={
        "prompts": PROMPTS,
        "max_new_tokens": MAX_NEW_TOKENS,
    },
    timeout=120,
)

response.raise_for_status()

batch_time = time.perf_counter() - start


# -------------------------
# Results
# -------------------------

print("\n--- Benchmark ---")
print(f"Individual: {individual_time:.2f}s")
print(f"Batch:      {batch_time:.2f}s")
print(f"Speedup:    {individual_time / batch_time:.2f}x")

print(
    f"Individual throughput: "
    f"{len(PROMPTS) / individual_time:.2f} req/s"
)

print(
    f"Batch throughput: "
    f"{len(PROMPTS) / batch_time:.2f} req/s"
)