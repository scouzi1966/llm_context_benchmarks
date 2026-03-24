#!/usr/bin/env python3
"""
Benchmark script for afm (macafm) — Apple Foundation Models and MLX models
served via the afm CLI (https://github.com/scouzi1966/maclocal-api).

The server exposes an OpenAI-compatible API at http://127.0.0.1:9999/v1 by default
and reports server-side timing metrics (prompt_time, completion_time, tok/s) in the
usage object, which this script extracts for accurate benchmarking.

Measurement notes (afm vs mlx engine parity):
  - prompt_tps / generation_tps: server-reported (Swift Date() wall-clock around
    MLX Swift generate), comparable to mlx engine's mlx_lm internal perf_counter.
  - TTFT: server prompt_time preferred (no HTTP overhead), matching mlx's derived
    safe_duration(prompt_tokens, prompt_tps). Wall-clock TTFT stored as wall_ttft.
  - Batch bs=1: server-reported per-request TPS (parity with mlx last_response).
    Batch bs>1: wall-clock aggregate (HTTP concurrent != single batched forward pass).
  - Peak memory: afm_profile.memory_peak_gib (MLX Swift Memory.snapshot().peakMemory),
    same underlying API as mlx's mx.get_peak_memory(). Includes server overhead.
  - total_time: wall-clock including HTTP overhead (~5-20ms fixed per request).

Two modes of operation:

  --managed (recommended for comparison with mlx engine):
    The script launches a fresh afm server per context size, then tears it down
    before the next size.  This mirrors how mlx_benchmark.py creates a fresh KV
    cache per run_benchmark() call.

  Without --managed:
    Connects to an already-running afm server (you manage the lifecycle).

Usage:
    # Managed mode — restart afm per context size
    python afm_benchmark.py mlx-community/Qwen3-0.6B-4bit --managed

    # With nightly binary
    python afm_benchmark.py mlx-community/Qwen3-0.6B-4bit --managed \\
        --afm-path /path/to/afm

    # External server mode (you start afm yourself)
    python afm_benchmark.py foundation
"""

import argparse
import asyncio
import json as json_mod
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp
import httpx

import benchmark_common as common

# ---------------------------------------------------------------------------
# Client helpers
# ---------------------------------------------------------------------------


def test_server_connection(base_url: str) -> bool:
    """Check that the afm server is reachable."""
    try:
        resp = httpx.get(f"{base_url}/models", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def wait_for_server(base_url: str, timeout: int = 120) -> bool:
    """Poll until the server responds or timeout is reached."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if test_server_connection(base_url):
            return True
        time.sleep(0.5)
    return False


def detect_framework(model_name: str) -> tuple[str, str]:
    """Determine framework label and directory name based on model name.

    Returns:
        Tuple of (display_name, dir_name) e.g. ("AFM", "afm") or ("AFM MLX", "afm_mlx")
    """
    if model_name.lower() == "foundation":
        return "AFM", "afm"
    return "AFM MLX", "afm_mlx"


# ---------------------------------------------------------------------------
# Managed server lifecycle
# ---------------------------------------------------------------------------


def context_file_tokens(context_file: Path) -> int:
    """Derive approximate token count from context filename (e.g. '2k' -> 2000)."""
    stem = context_file.stem  # e.g. "2k", "0.5k"
    return int(float(stem.rstrip("k")) * 1000)


def start_afm_server(
    afm_path: str,
    model: str,
    port: int,
    kv_bits: Optional[int] = None,
    concurrent: Optional[int] = None,
    extra_args: Optional[List[str]] = None,
) -> subprocess.Popen:
    """Launch an afm mlx server as a subprocess."""
    cmd = [afm_path, "mlx", "-m", model, "--port", str(port)]
    if kv_bits is not None:
        cmd.extend(["--kv-bits", str(kv_bits)])
    if concurrent is not None:
        cmd.extend(["--concurrent", str(concurrent)])
    if extra_args:
        cmd.extend(extra_args)
    print(f"  Starting afm: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc


def stop_afm_server(proc: subprocess.Popen) -> None:
    """Gracefully terminate an afm server subprocess."""
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


def _warmup_request(base_url: str, model: str) -> None:
    """Multi-round warmup to trigger Metal shader JIT for various sequence lengths.

    afm's MLX subcommand starts with prewarmEnabled=false, so the first inference
    at each sequence length pays shader compilation cost.  This warmup covers
    short, medium, and long sequences to pre-compile all relevant kernels,
    matching the reference benchmark_afm_vs_mlxlm.py warmup pattern.
    """
    url = f"{base_url}/chat/completions"
    warmup_specs = [
        ("Say hello.", 16),
        ("Write a paragraph about the weather today.", 128),
        ("Write a detailed paragraph about the history of computers.", 512),
        ("Write a long essay about artificial intelligence and its impact on society.", 1024),
        ("Explain quantum computing in detail.", 512),
    ]
    n = len(warmup_specs)
    for i, (prompt, max_tokens) in enumerate(warmup_specs):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": False,
        }
        try:
            resp = httpx.post(url, json=payload, timeout=120)
            tokens = resp.json().get("usage", {}).get("completion_tokens", "?")
            print(f"    warmup {i + 1}/{n}: {tokens} tokens (max {max_tokens})")
        except Exception as e:
            print(f"    warmup {i + 1}/{n} failed: {e}")


# ---------------------------------------------------------------------------
# Benchmark runner (httpx non-streaming with X-AFM-Profile)
# ---------------------------------------------------------------------------


def run_benchmark(
    base_url: str,
    model: str,
    context_file: Path,
    max_tokens: int = 200,
    timeout: int = 3600,
) -> Optional[Dict]:
    """Benchmark a single context file against afm.

    Uses non-streaming requests with X-AFM-Profile header. Server-reported
    timing (prompt_time, completion_time, TPS) comes from the usage object
    in the response body — measured at the same layer as mlx_lm's internal
    perf_counter timing. Non-streaming avoids SSE framing overhead that
    penalizes generation TPS in streaming mode.

    Returns a result dict on success, None on failure.
    """
    with open(context_file) as f:
        prompt = f.read()

    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
    }
    headers = {}

    start_time = time.perf_counter()

    try:
        resp = httpx.post(url, json=payload, headers=headers, timeout=timeout)
        if resp.status_code != 200:
            print(f"Error: HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        data = resp.json()
    except Exception as e:
        print(f"Error during benchmark: {e}")
        return None

    end_time = time.perf_counter()
    total_time = end_time - start_time

    # Extract usage
    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    server_prompt_time = usage.get("prompt_time")
    server_completion_time = usage.get("completion_time")
    server_prompt_tps = usage.get("prompt_tokens_per_second")
    server_gen_tps = usage.get("completion_tokens_per_second")
    ptd = usage.get("prompt_tokens_details") or {}
    cached_tokens = ptd.get("cached_tokens", 0)

    # Peak memory from usage (lightweight MLX allocator counter, no profiling overhead)
    peak_memory_gb = usage.get("peak_memory_gib", 0.0)

    # Extract generated text
    choices = data.get("choices", [])
    generated_text = ""
    if choices:
        msg = choices[0].get("message", {})
        generated_text = msg.get("content") or msg.get("reasoning_content") or ""

    # Fallback token counts
    if prompt_tokens == 0:
        prompt_tokens = len(prompt.split())
    if completion_tokens == 0:
        completion_tokens = len(generated_text.split())

    # --- Primary metrics from server; wall-clock total_time as verification ---
    timing_source = "server" if server_prompt_time is not None else "wall-clock"

    if server_prompt_time is not None:
        ttft = server_prompt_time
        prompt_eval_duration = server_prompt_time
        prompt_tps = (
            server_prompt_tps
            if server_prompt_tps
            else (prompt_tokens / prompt_eval_duration if prompt_eval_duration > 0 else 0.0)
        )
    else:
        ttft = 0.0
        prompt_eval_duration = 0.0
        prompt_tps = 0.0

    if server_completion_time is not None:
        eval_duration = server_completion_time
        generation_tps = (
            server_gen_tps if server_gen_tps else (completion_tokens / eval_duration if eval_duration > 0 else 0.0)
        )
    else:
        eval_duration = 0.0
        generation_tps = 0.0

    print(f"  Timing source:      {timing_source}")
    print(f"  Prompt tokens:      {prompt_tokens}" + (f" ({cached_tokens} cached)" if cached_tokens else ""))
    print(f"  Completion tokens:  {completion_tokens}")
    print(f"  TTFT:               {ttft:.3f}s")
    print(f"  Prompt eval:        {prompt_eval_duration:.3f}s  ({prompt_tps:.1f} t/s)")
    print(f"  Generation:         {eval_duration:.3f}s  ({generation_tps:.1f} t/s)")
    print(f"  Peak memory:        {peak_memory_gb:.1f} GB")
    print(f"  Total time:         {total_time:.2f}s (wall)")

    result = {
        "context_size": context_file.stem,
        "prompt_tokens": prompt_tokens,
        "prompt_tps": prompt_tps,
        "generation_tokens": completion_tokens,
        "generation_tps": generation_tps,
        "peak_memory_gb": peak_memory_gb,
        "total_time": total_time,
        "eval_duration": eval_duration,
        "prompt_eval_duration": prompt_eval_duration,
        "time_to_first_token": ttft,
        "timing_source": timing_source,
        "generated_text": generated_text,
    }
    if cached_tokens:
        result["cached_tokens"] = cached_tokens

    return result


# ---------------------------------------------------------------------------
# Batch benchmark (concurrent HTTP requests via aiohttp)
# ---------------------------------------------------------------------------


async def _send_one(session: aiohttp.ClientSession, url: str, model: str, prompt: str, gen_tokens: int) -> Dict:
    """Send one streaming request with X-AFM-Profile, parse SSE, return metrics."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen_tokens,
        "temperature": 0.7,
        "stream": True,
    }
    headers = {}
    prompt_tokens = 0
    completion_tokens = 0
    peak_memory_gb = 0.0
    server_prompt_tps = None
    server_gen_tps = None
    n_chunks = 0

    async with session.post(url, json=payload, headers=headers) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"HTTP {resp.status}: {body[:200]}")

        buf = ""
        async for raw in resp.content:
            buf += raw.decode()
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json_mod.loads(data)
                    if "afm_profile" in chunk:
                        peak_memory_gb = chunk["afm_profile"].get("memory_peak_gib", 0.0)
                        continue
                    usage = chunk.get("usage")
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                        completion_tokens = usage.get("completion_tokens", completion_tokens)
                        server_prompt_tps = usage.get("prompt_tokens_per_second")
                        server_gen_tps = usage.get("completion_tokens_per_second")
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        if delta.get("content") or delta.get("reasoning_content"):
                            n_chunks += 1
                except (json_mod.JSONDecodeError, KeyError):
                    continue

    # Prefer server-reported usage; fall back to chunk count
    if completion_tokens == 0:
        completion_tokens = n_chunks

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "peak_memory_gb": peak_memory_gb,
        "server_prompt_tps": server_prompt_tps,
        "server_gen_tps": server_gen_tps,
    }


async def _run_batch_trial(
    base_url: str, model: str, prompt: str, bs: int, gen_tokens: int, timeout: int
) -> tuple[List[Dict], int, float]:
    """Run one trial of bs concurrent requests. Returns (responses, failed_count, wall_time)."""
    url = f"{base_url}/chat/completions"
    conn = aiohttp.TCPConnector(limit=bs + 4)
    client_timeout = aiohttp.ClientTimeout(total=timeout)

    async with aiohttp.ClientSession(timeout=client_timeout, connector=conn) as session:
        start = time.perf_counter()
        tasks = [_send_one(session, url, model, prompt, gen_tokens) for _ in range(bs)]
        raw = await asyncio.gather(*tasks, return_exceptions=True)
        wall_time = time.perf_counter() - start

    responses = []
    failed = 0
    for r in raw:
        if isinstance(r, Exception):
            failed += 1
            if failed == 1:
                print(f"    Request failed: {r}")
        else:
            responses.append(r)

    return responses, failed, wall_time


def run_batch_benchmark(
    base_url: str,
    model: str,
    batch_sizes: List[int],
    prompt_tokens: int = 2048,
    gen_tokens: int = 256,
    num_trials: int = 3,
    timeout: int = 3600,
) -> List[Dict]:
    """Concurrent request throughput benchmark using aiohttp + asyncio.gather.

    Fires N simultaneous requests per trial, measures aggregate throughput.
    Each trial creates and tears down its own aiohttp session so connections
    are fully released between trials.
    """
    synthetic_prompt = ("The quick brown fox jumps over the lazy dog. " * (prompt_tokens // 10))[: prompt_tokens * 4]

    results = []
    for bs in batch_sizes:
        print(f"\n  Concurrency {bs} ({num_trials} trials, ~{prompt_tokens} prompt tokens, {gen_tokens} gen tokens)...")

        # Warmup
        print("    Warmup...")
        try:
            asyncio.run(_run_batch_trial(base_url, model, synthetic_prompt, 1, gen_tokens, timeout))
        except Exception:
            pass

        trial_prompt_tps = []
        trial_gen_tps = []
        trial_peak_mems = []

        for trial in range(num_trials):
            responses, failed, wall_time = asyncio.run(
                _run_batch_trial(base_url, model, synthetic_prompt, bs, gen_tokens, timeout)
            )

            if not responses:
                print(f"    Trial {trial + 1}: all requests failed, stopping batch size {bs}")
                break

            total_prompt = sum(r["prompt_tokens"] for r in responses)
            total_gen = sum(r["completion_tokens"] for r in responses)
            trial_peak_mem = max((r.get("peak_memory_gb", 0) for r in responses), default=0)

            # bs=1: use server-reported TPS (parity with mlx's last_response.prompt_tps)
            # bs>1: use wall-clock aggregate (concurrent HTTP ≠ single batched forward pass)
            if bs == 1 and responses[0].get("server_prompt_tps") is not None:
                trial_pp = responses[0]["server_prompt_tps"]
                trial_tg = responses[0]["server_gen_tps"] or 0
                src = "server"
            else:
                trial_pp = total_prompt / wall_time if wall_time > 0 else 0
                trial_tg = total_gen / wall_time if wall_time > 0 else 0
                src = "wall"

            if failed:
                print(
                    f"    Trial {trial + 1}: pp {trial_pp:.1f} tg {trial_tg:.1f} t/s "
                    f"({src}, wall {wall_time:.2f}s, {failed}/{bs} failed)"
                )
            else:
                print(f"    Trial {trial + 1}: pp {trial_pp:.1f} tg {trial_tg:.1f} t/s ({src}, wall {wall_time:.2f}s)")

            trial_prompt_tps.append(trial_pp)
            trial_gen_tps.append(trial_tg)
            trial_peak_mems.append(trial_peak_mem)

        if trial_prompt_tps:
            avg_pp = statistics.mean(trial_prompt_tps)
            avg_tg = statistics.mean(trial_gen_tps)
            peak_mem = max(trial_peak_mems) if trial_peak_mems else 0
            print(f"  Avg: pp {avg_pp:.1f} tg {avg_tg:.1f} t/s, peak mem {peak_mem:.1f} GB")

            results.append(
                {
                    "batch_size": bs,
                    "prompt_tps": round(avg_pp, 2),
                    "generation_tps": round(avg_tg, 2),
                    "peak_memory_gb": round(peak_mem, 3),
                }
            )

    return results


# ---------------------------------------------------------------------------
# Perplexity (via mlx_lm — same model weights, same result)
# ---------------------------------------------------------------------------


def compute_perplexity(model_name: str) -> tuple[Optional[float], Optional[Dict]]:
    """Compute perplexity using mlx_lm directly.

    Perplexity is an intrinsic model property — same weights give the same
    result regardless of serving layer (afm vs mlx_lm).
    """
    try:
        import mlx.core as mx
        import numpy as np
        from mlx_lm import load
        from mlx_lm.perplexity import eval_ppl, load_data
    except ImportError:
        print("  mlx-lm not available, skipping perplexity")
        return None, None

    print(f"  Loading model for perplexity: {model_name} ...")
    model, tokenizer = load(model_name)

    np.random.seed(123)
    mx.random.seed(123)

    ppl_num_samples = 256
    ppl_seq_length = 512
    ppl_dataset = "allenai/tulu-3-sft-mixture"

    print(f"  Dataset: {ppl_dataset} ({ppl_num_samples} samples, seq_length={ppl_seq_length})")
    data = load_data(tokenizer, ppl_dataset, num_samples=ppl_num_samples, sequence_length=ppl_seq_length)
    ppl, ppl_se = eval_ppl(model, data, batch_size=8)

    perplexity = float(ppl)
    perplexity_data = {
        "perplexity": perplexity,
        "std_error": float(ppl_se),
        "dataset": ppl_dataset,
        "num_samples": ppl_num_samples,
        "sequence_length": ppl_seq_length,
    }
    print(f"  Perplexity: {perplexity:.2f} (+/-{float(ppl_se):.2f})")

    # Free model memory
    del model, tokenizer

    return perplexity, perplexity_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_managed(args, base_url: str, context_files: List[Path], framework: str, output_dir: Path) -> List[Dict]:
    """Run benchmarks in managed mode — restart afm per context size."""
    results = []
    for ctx_file in context_files:
        print(f"\n{'=' * 60}")
        print(f"Context {ctx_file.name}")
        print(f"{'=' * 60}")

        proc = start_afm_server(
            args.afm_path,
            args.model,
            args.port,
            kv_bits=args.kv_bits,
            extra_args=args.afm_args.split() if args.afm_args else None,
        )
        try:
            print("  Waiting for server to be ready ...")
            if not wait_for_server(base_url, timeout=120):
                print(f"  ERROR: afm did not become ready within 120s, skipping {ctx_file.name}")
                continue

            # Warmup: first inference triggers Metal shader JIT compilation
            print("  Warmup (Metal JIT) ...")
            _warmup_request(base_url, args.model)

            result = run_benchmark(base_url, args.model, ctx_file, args.max_tokens, args.timeout)
            if result:
                results.append(result)
                if args.save_responses:
                    resp_path = output_dir / f"response_{result['context_size']}.txt"
                    common.save_generated_text(result, args.model, resp_path, framework)
        finally:
            print("  Stopping afm server ...")
            stop_afm_server(proc)

    return results


def run_external(args, base_url: str, context_files: List[Path], framework: str, output_dir: Path) -> List[Dict]:
    """Run benchmarks against an externally-managed afm server."""
    # Warmup run
    warmup_file = common.find_warmup_file()
    if warmup_file:
        print(f"\n{'=' * 50}")
        print(f"Warmup run (excluded from results): {warmup_file.name}")
        print(f"{'=' * 50}")
        run_benchmark(base_url, args.model, warmup_file, args.max_tokens, args.timeout)
        print("Warmup complete.")
    else:
        print("Warning: 0.5k.txt not found, skipping warmup.")

    results = []
    for ctx_file in context_files:
        print(f"\n{'=' * 50}")
        print(f"Benchmarking {ctx_file.name} ...")
        print(f"{'=' * 50}")

        result = run_benchmark(base_url, args.model, ctx_file, args.max_tokens, args.timeout)
        if result:
            results.append(result)
            if args.save_responses:
                resp_path = output_dir / f"response_{result['context_size']}.txt"
                common.save_generated_text(result, args.model, resp_path, framework)

    return results


def main() -> int:
    """Entry point for afm benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark afm — Apple Foundation Models and MLX models")
    parser.add_argument(
        "model",
        help="Model name: 'foundation' for Apple FM, or HF model ID for MLX (e.g., mlx-community/Qwen3-0.6B-4bit)",
    )
    parser.add_argument(
        "--afm-path",
        default="afm",
        help="Path to afm binary (default: afm, uses PATH)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL of the afm server (default: http://127.0.0.1:9999/v1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9999,
        help="Port for afm server (default: 9999). Ignored if --base-url is set.",
    )

    # Managed mode
    parser.add_argument(
        "--managed",
        action="store_true",
        help="Launch a fresh afm server per context size "
        "(recommended for apples-to-apples comparison with the mlx engine)",
    )
    parser.add_argument(
        "--kv-bits",
        type=int,
        default=None,
        help="Quantize KV cache (4 or 8 bits), passed to afm in managed mode",
    )
    parser.add_argument(
        "--afm-args",
        default=None,
        help="Extra flags passed to afm (e.g., '--max-kv-size 32000 --no-think')",
    )

    # Batch benchmark
    parser.add_argument(
        "--batch-sizes",
        default="1,2,4,8,16,32",
        help="Comma-separated concurrency levels for batch benchmark (default: 1,2,4,8,16,32)",
    )
    parser.add_argument(
        "--batch-prompt-tokens",
        type=int,
        default=2048,
        help="Approximate prompt tokens per request in batch benchmark (default: 2048)",
    )
    parser.add_argument(
        "--batch-gen-tokens",
        type=int,
        default=256,
        help="Tokens to generate per request in batch benchmark (default: 256)",
    )
    parser.add_argument(
        "--batch-trials",
        type=int,
        default=3,
        help="Trials per concurrency level, takes mean (default: 3)",
    )
    parser.add_argument("--no-batch", action="store_true", help="Skip batch benchmark")

    # Perplexity
    parser.add_argument("--no-perplexity", action="store_true", help="Skip perplexity computation")

    common.setup_common_args(parser)
    args = parser.parse_args()

    # Resolve base URL
    base_url = args.base_url if args.base_url else f"http://127.0.0.1:{args.port}/v1"
    base_url = base_url.rstrip("/")

    # In external mode, verify the server is reachable upfront
    if not args.managed:
        print(f"Testing connection to {base_url} ...")
        if not test_server_connection(base_url):
            print(f"Error: Cannot reach afm server at {base_url}")
            print("Make sure the server is running:")
            print("  afm                                          # Foundation model")
            print("  afm mlx -m mlx-community/Qwen3-0.6B-4bit    # MLX model")
            print("\nOr use --managed to let the benchmark manage the server lifecycle.")
            return 1
        print("Connected successfully.")

    framework, framework_dir = detect_framework(args.model)

    hardware_info = common.get_hardware_info()
    hardware_str = common.format_hardware_string(hardware_info)

    print(f"\nafm ({framework}) Benchmark" + (" [managed]" if args.managed else ""))
    print(f"Server:     {base_url}")
    print(f"Model:      {args.model}")
    print(f"Hardware:   {hardware_str}")
    print(f"Max tokens: {args.max_tokens}")
    if args.afm_path != "afm":
        print(f"Binary:     {args.afm_path}")
    if args.managed:
        print(f"Mode:       managed (restart per context)")
        if args.kv_bits:
            print(f"KV bits:    {args.kv_bits}")

    context_files = common.find_context_files(args.contexts)
    if not context_files:
        return 1

    output_dir = common.create_output_directory(framework_dir, args.model)

    # --- Perplexity (before context sweep, same order as mlx_benchmark) ---
    perplexity = None
    perplexity_data = None
    if args.no_perplexity:
        print("\nSkipping perplexity (--no-perplexity)")
    elif args.model.lower() == "foundation":
        print("\nSkipping perplexity (not supported for Foundation model)")
    else:
        print("\nComputing perplexity...")
        try:
            perplexity, perplexity_data = compute_perplexity(args.model)
        except Exception as e:
            print(f"Perplexity computation failed (continuing): {e}")

    # --- Batch benchmark ---
    batch_results = None
    if args.no_batch:
        print("\nSkipping batch benchmark (--no-batch)")
    elif args.model.lower() == "foundation":
        print("\nSkipping batch benchmark (not supported for Foundation model)")
    else:
        batch_sizes = [int(s.strip()) for s in args.batch_sizes.split(",")]
        print(f"\nRunning batch benchmark (concurrency levels: {batch_sizes})...")

        if args.managed:
            # Restart afm per concurrency level for clean state
            batch_results = []
            for bs in batch_sizes:
                concurrent_slots = bs * 2 + 4
                proc = start_afm_server(
                    args.afm_path,
                    args.model,
                    args.port,
                    kv_bits=args.kv_bits,
                    concurrent=concurrent_slots,
                    extra_args=args.afm_args.split() if args.afm_args else None,
                )
                print(f"  (concurrency {bs}, --concurrent {concurrent_slots})")
                try:
                    if not wait_for_server(base_url, timeout=120):
                        print(f"  ERROR: afm did not become ready for concurrency {bs}")
                        continue
                    level_results = run_batch_benchmark(
                        base_url,
                        args.model,
                        [bs],
                        prompt_tokens=args.batch_prompt_tokens,
                        gen_tokens=args.batch_gen_tokens,
                        num_trials=args.batch_trials,
                    )
                    batch_results.extend(level_results)
                finally:
                    print(f"  Stopping afm server (concurrency {bs}) ...")
                    stop_afm_server(proc)
            batch_results = batch_results or None
        else:
            # External mode — just run against the existing server
            try:
                batch_results = run_batch_benchmark(
                    base_url,
                    args.model,
                    batch_sizes,
                    prompt_tokens=args.batch_prompt_tokens,
                    gen_tokens=args.batch_gen_tokens,
                    num_trials=args.batch_trials,
                )
            except Exception as e:
                print(f"Batch benchmark failed (continuing): {e}")

        if batch_results:
            print(f"\nBatch benchmark complete: {len(batch_results)} concurrency levels tested")

    # --- Context sweep ---
    benchmark_start = time.perf_counter()

    if args.managed:
        results = run_managed(args, base_url, context_files, framework, output_dir)
    else:
        results = run_external(args, base_url, context_files, framework, output_dir)

    total_benchmark_time = time.perf_counter() - benchmark_start

    if not results:
        print("\nNo successful benchmark results.")
        return 1

    common.save_all_outputs(
        results,
        output_dir,
        args.model,
        framework,
        hardware_info,
        args,
        include_memory=True,
        perplexity=perplexity,
        perplexity_data=perplexity_data,
        batch_results=batch_results,
    )
    common.print_benchmark_summary(
        results,
        args.model,
        framework,
        hardware_info,
        output_dir,
        total_benchmark_time,
        perplexity=perplexity,
        batch_results=batch_results,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
