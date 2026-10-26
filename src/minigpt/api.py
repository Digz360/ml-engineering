from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dataclasses import dataclass
from queue import Queue
import time

from src.minigpt.model_loader import load_minigpt
import torch.nn.functional as F

import os
import threading

MAX_CONCURRENT_INFERENCE = 10
MAX_REQUESTS_IN_SYSTEM = 30
MAX_BATCH_SIZE = 4

request_queue = Queue()
inference_semaphore = threading.Semaphore(MAX_CONCURRENT_INFERENCE)
admission_semaphore = threading.BoundedSemaphore(MAX_REQUESTS_IN_SYSTEM)

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "models/minigpt_large_v2_sft_best.pt"
)

TOKENIZER_PATH = os.getenv(
    "TOKENIZER_PATH",
    "models/tokenizer_large.json"
)

model = None
tokenizer = None
device = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, device

    print("Loading MiniGPT...")

    model, tokenizer, checkpoint, device = load_minigpt(
        MODEL_PATH,
        TOKENIZER_PATH,
    )

    print(f"MiniGPT loaded on {device}")

    scheduler_thread = threading.Thread(
        target=inference_scheduler,
        daemon=True,
    )

    scheduler_thread.start()

    yield


app = FastAPI(
    title="MiniGPT API",
    lifespan=lifespan,
)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100

class BatchGenerateRequest(BaseModel):
    prompts: list[str]
    max_new_tokens: int = 20

@dataclass
class InferenceJob:
    prompt: str
    max_new_tokens: int
    done: threading.Event

    result: str | None = None
    error: Exception | None = None

    token_ids: list[int] | None = None
    generated_ids: list[int] | None = None

    past_key_values: list | None = None
    next_logits: torch.Tensor | None = None

    generated_count: int = 0

def initialize_job(job):

    prompt = (
        f"Utilizador: {job.prompt}\n"
        f"Assistente: "
    )

    job.token_ids = tokenizer.encode(prompt).ids
    job.generated_ids = []
    job.generated_count = 0

    context = torch.tensor(
        [job.token_ids],
        dtype=torch.long,
        device=device,
    )

    # PREFILL — prompt inteiro apenas UMA vez
    with torch.inference_mode():

        logits, job.past_key_values = model(
            context,
            use_cache=True,
        )

    job.next_logits = logits[:, -1, :]


def decode_one_step(active_jobs):
    eos_id = tokenizer.token_to_id("[EOS]")
    finished_jobs = []
    jobs_to_decode = []
    sampled_tokens = []

    # --------------------------------------------------
    # 1. Cada job já tem os logits para escolher
    #    o seu próximo token.
    # --------------------------------------------------
    for job in active_jobs:

        next_logits = job.next_logits / 0.5

        probs = torch.softmax(
            next_logits,
            dim=-1,
        )

        next_token = torch.multinomial(
            probs,
            num_samples=1,
        )

        token = next_token.item()

        # EOS -> acabou
        if token == eos_id:
            finished_jobs.append(job)
            continue

        job.token_ids.append(token)
        job.generated_ids.append(token)
        job.generated_count += 1

        # Atingiu limite pedido -> acabou
        if job.generated_count >= job.max_new_tokens:
            finished_jobs.append(job)
            continue

        # Este job precisa de continuar a gerar
        jobs_to_decode.append(job)
        sampled_tokens.append(token)

    # Todos acabaram neste step
    if not jobs_to_decode:
        return finished_jobs

    # --------------------------------------------------
    # 2. Descobrir tamanho real do cache de cada job
    # --------------------------------------------------
    cache_lengths = [
        job.past_key_values[0][0].shape[2]
        for job in jobs_to_decode
    ]

    max_cache_length = max(cache_lengths)

    # Proteção contra position embedding > block_size
    if max(cache_lengths) >= model.block_size:
        raise RuntimeError(
            "KV cache reached model block_size."
        )

    # --------------------------------------------------
    # 3. Juntar os tokens novos num batch
    #
    # [A]
    # [B]  -> [BATCH, 1]
    # [C]
    # --------------------------------------------------
    next_tokens = torch.tensor(
        sampled_tokens,
        dtype=torch.long,
        device=device,
    ).unsqueeze(1)

    # Posição REAL de cada token.
    #
    # Mesmo que façamos padding até max_cache_length,
    # cada sequência mantém a sua posição original.
    position_ids = torch.tensor(
        cache_lengths,
        dtype=torch.long,
        device=device,
    ).unsqueeze(1)

    # --------------------------------------------------
    # 4. Construir attention mask
    #
    # True  = token real, pode receber attention
    # False = padding, ignorar
    #
    # +1 porque o modelo vai acrescentar o token atual.
    # --------------------------------------------------
    batch_size = len(jobs_to_decode)

    attention_mask = torch.zeros(
        (
            batch_size,
            1,
            1,
            max_cache_length + 1,
        ),
        dtype=torch.bool,
        device=device,
    )

    for i, cache_length in enumerate(cache_lengths):

        padding = max_cache_length - cache_length

        # tokens antigos reais
        attention_mask[
            i,
            :,
            :,
            padding:max_cache_length
        ] = True

        # token novo
        attention_mask[
            i,
            :,
            :,
            max_cache_length
        ] = True

    # --------------------------------------------------
    # 5. Juntar KV caches dos jobs
    # --------------------------------------------------
    num_layers = len(jobs_to_decode[0].past_key_values)

    batched_past_key_values = []

    for layer_idx in range(num_layers):

        padded_keys = []
        padded_values = []

        for job, cache_length in zip(
            jobs_to_decode,
            cache_lengths,
        ):

            k, v = job.past_key_values[layer_idx]

            padding = max_cache_length - cache_length

            # Padding à esquerda na dimensão temporal
            if padding > 0:

                k = F.pad(
                    k,
                    (0, 0, padding, 0),
                    value=0.0,
                )

                v = F.pad(
                    v,
                    (0, 0, padding, 0),
                    value=0.0,
                )

            padded_keys.append(k)
            padded_values.append(v)

        # Cada cache tinha B=1.
        # Agora criamos B=número de jobs.
        batched_k = torch.cat(
            padded_keys,
            dim=0,
        )

        batched_v = torch.cat(
            padded_values,
            dim=0,
        )

        batched_past_key_values.append(
            (batched_k, batched_v)
        )

    # --------------------------------------------------
    # 6. UMA ÚNICA CHAMADA AO MODELO
    # --------------------------------------------------
    with torch.inference_mode():

        logits, batched_new_cache = model(
            next_tokens,
            past_key_values=batched_past_key_values,
            use_cache=True,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    # --------------------------------------------------
    # 7. Separar novamente os caches
    # --------------------------------------------------
    for i, (job, old_length) in enumerate(
        zip(jobs_to_decode, cache_lengths)
    ):

        padding = max_cache_length - old_length

        job_cache = []

        for layer_idx in range(num_layers):

            k_batch, v_batch = batched_new_cache[layer_idx]

            # Estrutura:
            #
            # [padding | cache antigo | token novo]
            #
            # Queremos:
            #
            # [cache antigo | token novo]

            k = k_batch[
                i:i + 1,
                :,
                padding:max_cache_length + 1,
                :
            ].contiguous()

            v = v_batch[
                i:i + 1,
                :,
                padding:max_cache_length + 1,
                :
            ].contiguous()

            job_cache.append((k, v))

        job.past_key_values = job_cache

        # logits correspondentes a ESTE job
        job.next_logits = logits[
            i:i + 1,
            -1,
            :
        ]

    return finished_jobs


def inference_scheduler():

    active_jobs = []

    while True:

        # 1. Se não há jobs ativos, esperar pelo primeiro
        if not active_jobs:
            job = request_queue.get()
            initialize_job(job)
            active_jobs.append(job)

        # 2. Preencher espaços vazios do batch
        while len(active_jobs) < MAX_BATCH_SIZE:
            try:
                job = request_queue.get_nowait()
                initialize_job(job)
                active_jobs.append(job)
            except:
                break

        print(
            "DECODE STEP | "
            f"Active batch: {len(active_jobs)} | "
            f"Queue: {request_queue.qsize()} | "
            f"Generated: {[job.generated_count for job in active_jobs]}"
        )

        # 3. Fazer UM decoding step para todo o batch
        finished_jobs = decode_one_step(active_jobs)

        # 4. Finalizar os jobs que acabaram
        for job in finished_jobs:

            job.result = tokenizer.decode(
                job.generated_ids
            )

            # Acordar o request HTTP correspondente
            job.done.set()

        # 5. Remover os terminados
        active_jobs = [
            job
            for job in active_jobs
            if job not in finished_jobs
        ]

        if finished_jobs:
            print(
                f"FINISHED: {len(finished_jobs)} | "
                f"Remaining active: {len(active_jobs)}"
            )


@app.post("/generate-batch")
def generate_batch(request: BatchGenerateRequest):

    formatted_prompts = [
        f"Utilizador: {prompt}\nAssistente: "
        for prompt in request.prompts
    ]

    encoded_prompts = [
        tokenizer.encode(prompt).ids
        for prompt in formatted_prompts
    ]

    max_length = max(
        len(tokens)
        for tokens in encoded_prompts
    )

    pad_id = tokenizer.token_to_id("[PAD]")

    padded_prompts = [
        [pad_id] * (max_length - len(tokens)) + tokens
        for tokens in encoded_prompts
    ]

    context = torch.tensor(
        padded_prompts,
        dtype=torch.long,
        device=device,
    )

    generated = context.clone()

    eos_id = tokenizer.token_to_id("[EOS]")

    finished = torch.zeros(
        context.shape[0],
        dtype=torch.bool,
        device=device,
    )

    with inference_semaphore:
        with torch.inference_mode():

            for _ in range(request.max_new_tokens):

                input_tokens = generated[:, -model.block_size:]

                logits = model(input_tokens)

                next_logits = logits[:, -1, :]

                temperature = 0.5
                next_logits = next_logits / temperature

                probs = torch.softmax(next_logits, dim=-1)

                next_tokens = torch.multinomial(
                    probs,
                    num_samples=1,
                )

                just_finished = next_tokens.squeeze(1) == eos_id

                finished = finished | just_finished

                generated = torch.cat(
                    [generated, next_tokens],
                    dim=1,
                )

                if finished.all():
                    break
    responses = []

    for i, prompt in enumerate(request.prompts):

        response_ids = generated[
            i,
            context.shape[1]:
        ].tolist()

        # Cortar tudo a partir do primeiro EOS
        if eos_id in response_ids:
            eos_position = response_ids.index(eos_id)
            response_ids = response_ids[:eos_position]

        response = tokenizer.decode(response_ids)

        responses.append({
            "prompt": prompt,
            "text": response,
        })

    return {
        "responses": responses
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(device),
    }

@app.post("/generate")
def generate(request: GenerateRequest):

    admitted = admission_semaphore.acquire(blocking=False)

    if not admitted:
        raise HTTPException(
            status_code=503,
            detail="Server overloaded. Try again later."
        )

    try:
        job = InferenceJob(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            done=threading.Event(),
        )

        # Colocar trabalho na fila
        request_queue.put(job)

        # Esperar que o scheduler termine este job
        job.done.wait()

        # Se ocorreu algum erro no scheduler
        if job.error is not None:
            raise HTTPException(
                status_code=500,
                detail=str(job.error),
            )

        return {
            "prompt": request.prompt,
            "text": job.result,
        }

    finally:
        admission_semaphore.release()