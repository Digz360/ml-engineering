import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "http://207.175.118.156/generate"
TOTAL_REQUESTS = 200
WORKERS = 20
TIMEOUT = 120
def send_request(i):
    try:
        r = requests.post(
            URL,
            json={
                "prompt": "Explica machine learning.",
                "use_rag": False,
                "max_new_tokens": 30
            },
            timeout=TIMEOUT
        )
        return r.status_code
    except requests.exceptions.Timeout:
        return "timeout"
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        return "error"


stats = {}
start = time.time()

with ThreadPoolExecutor(max_workers=WORKERS) as executor:

    futures = [
        executor.submit(send_request, i)
        for i in range(TOTAL_REQUESTS)
    ]

    for i, future in enumerate(as_completed(futures), 1):

        result = future.result()
        stats[result] = stats.get(result, 0) + 1

        if i % 1 == 0:
            elapsed = time.time() - start
            print(
                f"{i}/{TOTAL_REQUESTS} | "
                f"{i/elapsed:.2f} req/s | "
                f"{stats}"
            )

elapsed = time.time() - start

print("\nRESULTADOS")
print(stats)
print(f"Tempo: {elapsed:.1f}s")
print(f"Throughput: {TOTAL_REQUESTS/elapsed:.2f} req/s")