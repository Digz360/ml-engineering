import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


URL = "http://35.241.43.231/generate"

CONCURRENCY_LEVELS = [100]

PROMPT = "O que é machine learning?"
MAX_NEW_TOKENS = 100


def send_request():
    start = time.perf_counter()

    try:
        response = requests.post(
            URL,
            json={
                "prompt": PROMPT,
                "max_new_tokens": MAX_NEW_TOKENS,
                 "use_rag": False,
            },
            timeout=120,
        )

        latency = time.perf_counter() - start

        if response.status_code == 200:
            status = "success"
        elif response.status_code == 503:
            status = "rejected"
        else:
            print("HTTP ERROR:", response.status_code, response.text[:200])
            status = "error"

        return {
            "status": status,
            "latency": latency,
        }

    except requests.exceptions.Timeout:
        return {
            "status": "timeout",
            "latency": time.perf_counter() - start,
        }

    except Exception:
        return {
            "status": "error",
            "latency": time.perf_counter() - start,
        }


def percentile(values, percentile):
    values = sorted(values)

    index = (len(values) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)

    fraction = index - lower

    return values[lower] + (values[upper] - values[lower]) * fraction


def run_test(concurrency):
    print(f"\n--- Concurrency: {concurrency} ---")

    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(send_request)
            for _ in range(concurrency)
        ]

        results = [
            future.result()
            for future in as_completed(futures)
        ]

    total_time = time.perf_counter() - start

    successful = [r for r in results if r["status"] == "success"]
    rejected = [r for r in results if r["status"] == "rejected"]
    timeouts = [r for r in results if r["status"] == "timeout"]
    errors = [r for r in results if r["status"] == "error"]

    latencies = [r["latency"] for r in successful]

    if not latencies:
        print("All requests failed.")
        return

    throughput = len(successful) / total_time

    print(f"Successful: {len(successful)}/{concurrency}")
    print(f"Rejected (503): {len(rejected)}")
    print(f"Timeouts: {len(timeouts)}")
    print(f"Other errors: {len(errors)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average: {statistics.mean(latencies):.2f}s")
    print(f"p50: {percentile(latencies, 0.50):.2f}s")
    print(f"p95: {percentile(latencies, 0.95):.2f}s")
    print(f"p99: {percentile(latencies, 0.99):.2f}s")
    print(f"Throughput: {throughput:.2f} req/s")


if __name__ == "__main__":
    for concurrency in CONCURRENCY_LEVELS:
        run_test(concurrency)
