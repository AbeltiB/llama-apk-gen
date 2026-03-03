#!/usr/bin/env python3
"""
End-to-end stress tester for llama-apk-gen generation APIs.

Features
- Stress test with generated prompts (default 500) or a robust custom prompt file loader.
- Full API flow: /generate -> /task/{task_id} polling -> /results + /results/{task_id}/export.
- Validates export JSON shape.
- Saves JSON outputs for FE testing.
- Optionally downloads attachment payloads from ?download=true endpoint.
- Produces a machine-readable summary and a human-readable analysis report.

Backward-compatible alias for a common filename typo.

Use: python test/stres_test_system.py ...
Delegates to stress_test_system.py.
"""

from stress_test_system import execute, parse_args
import asyncio

if __name__ == "__main__":
    raise SystemExit(asyncio.run(execute(parse_args())))

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

REQUIRED_TOP_LEVEL_KEYS = {
    "importManager",
    "stateManager",
    "functionManager",
    "componentManager",
    "uiManager",
    "code",
    "blocklyManager",
    "blocklyByScreen",
}


@dataclass
class PromptRunResult:
    index: int
    prompt: str
    task_id: Optional[str]
    submitted: bool
    final_status: str
    success: bool
    output_valid: bool
    used_heuristic: Optional[bool]
    generation_method: Optional[str]
    provider_used: Optional[str]
    prompt_satisfied: Optional[bool]
    submit_ms: Optional[int]
    queue_to_done_ms: Optional[int]
    export_fetch_ms: Optional[int]
    export_download_ms: Optional[int]
    total_ms: int
    error_category: Optional[str]
    error_detail: Optional[str]
    output_file: Optional[str]
    download_file: Optional[str]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str, max_len: int = 60) -> str:
    v = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    return (v[:max_len] or "prompt").strip("_")


def ensure_output_dirs(base_dir: Path) -> Dict[str, Path]:
    """Create output folder structure if missing."""
    json_dir = base_dir / "outputs" / "json"
    download_dir = base_dir / "outputs" / "downloads"
    reports_dir = base_dir / "outputs" / "reports"
    for path in (json_dir, download_dir, reports_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {"json": json_dir, "downloads": download_dir, "reports": reports_dir}


def _normalize_prompts(prompts: List[str], dedupe: bool = True, min_len: int = 10) -> List[str]:
    cleaned: List[str] = []
    seen: set[str] = set()
    for raw in prompts:
        prompt = " ".join((raw or "").strip().split())
        if not prompt or prompt.startswith("#"):
            continue
        if len(prompt) < min_len:
            continue
        key = prompt.lower()
        if dedupe and key in seen:
            continue
        seen.add(key)
        cleaned.append(prompt)
    return cleaned


def load_prompts_from_file(path: Path, dedupe: bool = True, min_len: int = 10) -> List[str]:
    """Robust custom prompt loader.

    Supports:
    - .txt / .md (line-based, comments with #)
    - .json (array of strings OR {"prompts": [...]})
    - .csv (first column named prompt OR first column fallback)
    """
    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {path}")

    ext = path.suffix.lower()
    prompts: List[str] = []

    if ext in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8-sig")
        prompts = text.splitlines()
    elif ext == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(data, list):
            prompts = [str(item) for item in data]
        elif isinstance(data, dict) and isinstance(data.get("prompts"), list):
            prompts = [str(item) for item in data["prompts"]]
        else:
            raise ValueError("JSON prompts file must be a list of strings or {'prompts': [...]}.")
    elif ext == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as fp:
            reader = csv.DictReader(fp)
            if reader.fieldnames and "prompt" in [f.lower() for f in reader.fieldnames]:
                prompt_field = next(f for f in reader.fieldnames if f.lower() == "prompt")
                prompts = [row.get(prompt_field, "") for row in reader]
            else:
                fp.seek(0)
                rows = csv.reader(fp)
                prompts = [row[0] for row in rows if row]
    else:
        raise ValueError(f"Unsupported prompts file extension: {ext}")

    normalized = _normalize_prompts(prompts, dedupe=dedupe, min_len=min_len)
    if not normalized:
        raise ValueError("No valid prompts found after normalization (check file content/format).")
    return normalized


def generate_prompts(count: int) -> List[str]:
    app_types = [
        "counter", "todo", "calculator", "weather", "notes", "timer", "habit tracker",
        "budget tracker", "flashcards", "quiz", "inventory", "fitness log", "recipe",
        "chat", "bookmarks", "pomodoro", "journal", "contacts", "unit converter",
    ]
    audiences = ["students", "nurses", "drivers", "developers", "teachers", "parents"]
    constraints = [
        "dark mode only", "offline-first", "simple one-screen UI", "high-contrast accessibility",
        "large touch targets", "minimalistic style", "include loading state",
    ]
    features = [
        "add/edit/delete items", "search and filter", "reset button", "increment/decrement controls",
        "empty-state message", "error handling", "local persistence", "summary stats card",
        "toggle switches", "input validation", "export data button", "quick actions",
    ]

    prompts: List[str] = []
    i = 0
    while len(prompts) < count:
        app = app_types[i % len(app_types)]
        aud = audiences[(i // len(app_types)) % len(audiences)]
        c1 = constraints[(i * 3) % len(constraints)]
        f1 = features[(i * 5) % len(features)]
        f2 = features[(i * 7 + 2) % len(features)]
        prompts.append(
            f"Create a {app} mobile app for {aud}. "
            f"It should support {f1} and {f2}, with {c1}. "
            "Generate complete architecture, layout, and interaction logic."
        )
        i += 1
    return prompts


def validate_export_payload(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    missing = sorted(REQUIRED_TOP_LEVEL_KEYS - set(payload.keys()))
    if missing:
        errors.append(f"Missing top-level keys: {', '.join(missing)}")

    if "blocklyManager" in payload and isinstance(payload["blocklyManager"], dict):
        if "blocklyByScreen" in payload["blocklyManager"]:
            errors.append("blocklyByScreen is nested inside blocklyManager (must be top-level)")

    for key in ("componentManager", "uiManager", "stateManager"):
        if key in payload and not isinstance(payload[key], dict):
            errors.append(f"{key} should be an object")

    return errors


def evaluate_prompt_satisfaction(prompt: str, export_payload: Dict[str, Any]) -> bool:
    keywords = [w.lower() for w in re.findall(r"[a-zA-Z]{4,}", prompt)]
    stop = {
        "create", "mobile", "should", "with", "generate", "complete", "logic", "screen",
        "layout", "architecture", "app", "for", "include",
    }
    wanted = [k for k in keywords if k not in stop][:12]
    if not wanted:
        return True
    haystack = json.dumps(export_payload, ensure_ascii=False).lower()
    hits = sum(1 for word in wanted if word in haystack)
    return hits >= max(1, min(3, len(wanted) // 4))


async def run_one(
    client: httpx.AsyncClient,
    base_url: str,
    output_json_dir: Path,
    output_download_dir: Path,
    prompt: str,
    index: int,
    poll_interval: float,
    task_timeout_s: int,
    save_download: bool,
) -> PromptRunResult:
    start = time.perf_counter()
    task_id: Optional[str] = None
    submit_ms: Optional[int] = None
    queue_to_done_ms: Optional[int] = None
    export_fetch_ms: Optional[int] = None
    export_download_ms: Optional[int] = None

    try:
        t0 = time.perf_counter()
        submit_resp = await client.post(
            f"{base_url}/api/v1/generate",
            json={
                "prompt": prompt,
                "user_id": f"stress_user_{index}",
                "session_id": f"stress_session_{index}",
                "priority": 1,
            },
            timeout=30.0,
        )
        submit_ms = int((time.perf_counter() - t0) * 1000)
    except httpx.RequestError as exc:
        return PromptRunResult(index, prompt, None, False, "submit_connection_error", False, False, None, None, None, None, None, None, None, int((time.perf_counter() - start) * 1000), "connection", str(exc), None, None)

    if submit_resp.status_code >= 400:
        return PromptRunResult(index, prompt, None, False, f"submit_http_{submit_resp.status_code}", False, False, None, None, None, None, submit_ms, None, None, int((time.perf_counter() - start) * 1000), "submit_http", submit_resp.text[:800], None, None)

    task_id = submit_resp.json().get("task_id")
    if not task_id:
        return PromptRunResult(index, prompt, None, False, "submit_missing_task_id", False, False, None, None, None, None, submit_ms, None, None, int((time.perf_counter() - start) * 1000), "submit_schema", f"Missing task_id in response: {submit_resp.text[:500]}", None, None)

    poll_start = time.perf_counter()
    deadline = poll_start + task_timeout_s
    final_status = "processing"
    task_blob: Dict[str, Any] = {}

    while time.perf_counter() < deadline:
        try:
            poll_resp = await client.get(f"{base_url}/api/v1/task/{task_id}", timeout=20.0)
        except httpx.RequestError as exc:
            await asyncio.sleep(poll_interval)
            final_status = "poll_connection_error"
            task_blob = {"poll_error": str(exc)}
            continue

        if poll_resp.status_code >= 400:
            final_status = f"poll_http_{poll_resp.status_code}"
            task_blob = {"poll_http_body": poll_resp.text[:500]}
            await asyncio.sleep(poll_interval)
            continue

        task_blob = poll_resp.json()
        status = str(task_blob.get("status", "")).lower()
        final_status = status or final_status
        if status in {"completed", "failed", "cancelled"}:
            break
        await asyncio.sleep(poll_interval)

    queue_to_done_ms = int((time.perf_counter() - poll_start) * 1000)
    if final_status != "completed":
        category = "timeout" if time.perf_counter() >= deadline else "task_not_completed"
        return PromptRunResult(index, prompt, task_id, True, final_status, False, False, None, None, None, None, submit_ms, queue_to_done_ms, None, int((time.perf_counter() - start) * 1000), category, json.dumps(task_blob)[:800], None, None)

    generation_method = None
    provider_used = None
    used_heuristic: Optional[bool] = None
    try:
        result_resp = await client.get(f"{base_url}/api/v1/results/{task_id}", timeout=30.0)
        if result_resp.status_code < 400:
            result_json = result_resp.json()
            metadata = result_json.get("metadata") if isinstance(result_json, dict) else {}
            if isinstance(metadata, dict):
                generation_method = metadata.get("generation_method")
                provider_used = metadata.get("provider_used")
                used_heuristic = metadata.get("heuristic_fallback_used")
    except httpx.RequestError:
        pass

    try:
        t1 = time.perf_counter()
        export_resp = await client.get(f"{base_url}/api/v1/results/{task_id}/export", timeout=30.0)
        export_fetch_ms = int((time.perf_counter() - t1) * 1000)
    except httpx.RequestError as exc:
        return PromptRunResult(index, prompt, task_id, True, "export_connection_error", False, False, used_heuristic, generation_method, provider_used, None, submit_ms, queue_to_done_ms, None, int((time.perf_counter() - start) * 1000), "connection", str(exc), None, None)

    if export_resp.status_code >= 400:
        return PromptRunResult(index, prompt, task_id, True, f"export_http_{export_resp.status_code}", False, False, used_heuristic, generation_method, provider_used, None, submit_ms, queue_to_done_ms, export_fetch_ms, int((time.perf_counter() - start) * 1000), "export_http", export_resp.text[:800], None, None)

    try:
        export_payload = export_resp.json()
    except Exception as exc:  # noqa: BLE001
        return PromptRunResult(index, prompt, task_id, True, "export_invalid_json", False, False, used_heuristic, generation_method, provider_used, None, submit_ms, queue_to_done_ms, export_fetch_ms, int((time.perf_counter() - start) * 1000), "export_schema", str(exc), None, None)

    errors = validate_export_payload(export_payload)
    valid = not errors

    file_stem = f"{index:04d}_{slugify(prompt)}_{task_id}"
    output_file = output_json_dir / f"{file_stem}.json"
    output_file.write_text(json.dumps(export_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    download_file_str: Optional[str] = None
    if save_download:
        try:
            t2 = time.perf_counter()
            dl_resp = await client.get(f"{base_url}/api/v1/results/{task_id}/export?download=true", timeout=30.0)
            export_download_ms = int((time.perf_counter() - t2) * 1000)
            if dl_resp.status_code < 400:
                download_path = output_download_dir / f"{file_stem}_download.json"
                download_path.write_bytes(dl_resp.content)
                download_file_str = str(download_path)
            else:
                valid = False
                errors.append(f"download_endpoint_http_{dl_resp.status_code}")
        except httpx.RequestError as exc:
            valid = False
            errors.append(f"download_connection_error: {exc}")

    return PromptRunResult(
        index=index,
        prompt=prompt,
        task_id=task_id,
        submitted=True,
        final_status="completed",
        success=valid,
        output_valid=valid,
        used_heuristic=used_heuristic,
        generation_method=generation_method,
        provider_used=provider_used,
        prompt_satisfied=evaluate_prompt_satisfaction(prompt, export_payload),
        submit_ms=submit_ms,
        queue_to_done_ms=queue_to_done_ms,
        export_fetch_ms=export_fetch_ms,
        export_download_ms=export_download_ms,
        total_ms=int((time.perf_counter() - start) * 1000),
        error_category=None if valid else "output_validation",
        error_detail=None if valid else "; ".join(errors),
        output_file=str(output_file),
        download_file=download_file_str,
    )


def percentile(values: List[int], p: float) -> Optional[float]:
    if not values:
        return None
    vals = sorted(values)
    idx = int((len(vals) - 1) * p)
    return float(vals[idx])


def build_markdown_report(
    run_id: str,
    base_url: str,
    prompt_count: int,
    started_at: str,
    finished_at: str,
    results: List[PromptRunResult],
) -> str:
    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]
    connection_failures = [r for r in results if r.error_category == "connection"]
    heuristic_used = [r for r in results if r.used_heuristic is True]
    satisfied = [r for r in results if r.prompt_satisfied is True]

    total_times = [r.total_ms for r in results]
    done_times = [r.queue_to_done_ms for r in results if r.queue_to_done_ms is not None]
    download_times = [r.export_download_ms for r in results if r.export_download_ms is not None]

    failure_by_reason: Dict[str, int] = {}
    for r in failures:
        key = r.error_category or r.final_status
        failure_by_reason[key] = failure_by_reason.get(key, 0) + 1

    top_fail_lines = "\n".join(
        f"- **{k}**: {v}" for k, v in sorted(failure_by_reason.items(), key=lambda kv: kv[1], reverse=True)
    ) or "- None"

    return f"""# Stress Test Analysis Report

- **Run ID:** `{run_id}`
- **Base URL:** `{base_url}`
- **Prompts executed:** {prompt_count}
- **Started (UTC):** {started_at}
- **Finished (UTC):** {finished_at}

## Overall Health
- **Successful + valid outputs:** {len(successes)}/{len(results)} ({(len(successes)/len(results)*100) if results else 0:.2f}%)
- **Failed runs:** {len(failures)}
- **Connection-related failures:** {len(connection_failures)}
- **Heuristic fallback used:** {len(heuristic_used)}
- **Prompt satisfaction (heuristic):** {len(satisfied)}/{len(results)} ({(len(satisfied)/len(results)*100) if results else 0:.2f}%)

## Performance
- **Total time p50:** {percentile(total_times, 0.50)} ms
- **Total time p95:** {percentile(total_times, 0.95)} ms
- **Task completion time p50:** {percentile(done_times, 0.50)} ms
- **Task completion time p95:** {percentile(done_times, 0.95)} ms
- **Download endpoint time p50:** {percentile(download_times, 0.50)} ms
- **Download endpoint time p95:** {percentile(download_times, 0.95)} ms

## Failure Breakdown
{top_fail_lines}

## Why these fields matter
- `connection` errors usually mean infra/network instability.
- `timeout` usually indicates queue saturation or slow upstream providers.
- `export_http`/`export_schema` indicates backend output delivery or serialization issues.
- `output_validation` indicates structurally invalid exported JSON.
"""


async def execute(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parent
    dirs = ensure_output_dirs(root)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    started_at = utc_now()

    if args.prompts_file:
        prompts = load_prompts_from_file(Path(args.prompts_file), dedupe=not args.no_dedupe, min_len=args.min_prompt_length)
    else:
        prompts = _normalize_prompts(generate_prompts(args.count), dedupe=not args.no_dedupe, min_len=args.min_prompt_length)

    if args.max_prompts:
        prompts = prompts[: args.max_prompts]

    if args.dry_run:
        print(f"[dry-run] Prepared {len(prompts)} prompts. No network calls made.")
        return 0

    results: List[PromptRunResult] = []
    sem = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient() as client:
        async def _runner(idx: int, prompt: str) -> None:
            async with sem:
                res = await run_one(
                    client=client,
                    base_url=args.base_url.rstrip("/"),
                    output_json_dir=dirs["json"],
                    output_download_dir=dirs["downloads"],
                    prompt=prompt,
                    index=idx,
                    poll_interval=args.poll_interval,
                    task_timeout_s=args.task_timeout,
                    save_download=args.download_exports,
                )
                results.append(res)
                print(f"[{idx + 1}/{len(prompts)}] {res.final_status} valid={res.output_valid} task={res.task_id}")

        await asyncio.gather(*[_runner(i, p) for i, p in enumerate(prompts)])

    results = sorted(results, key=lambda r: r.index)
    finished_at = utc_now()

    summary_json_path = dirs["reports"] / f"stress_summary_{run_id}.json"
    summary_md_path = dirs["reports"] / f"stress_analysis_{run_id}.md"

    summary_payload = {
        "run_id": run_id,
        "base_url": args.base_url,
        "started_at": started_at,
        "finished_at": finished_at,
        "prompt_count": len(prompts),
        "config": {
            "count": args.count,
            "max_prompts": args.max_prompts,
            "concurrency": args.concurrency,
            "poll_interval": args.poll_interval,
            "task_timeout": args.task_timeout,
            "prompts_file": args.prompts_file,
            "download_exports": args.download_exports,
            "min_prompt_length": args.min_prompt_length,
            "dedupe": not args.no_dedupe,
        },
        "results": [asdict(r) for r in results],
    }
    summary_json_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    report = build_markdown_report(
        run_id=run_id,
        base_url=args.base_url,
        prompt_count=len(prompts),
        started_at=started_at,
        finished_at=finished_at,
        results=results,
    )
    summary_md_path.write_text(report, encoding="utf-8")

    valid_count = sum(1 for r in results if r.output_valid)
    print("\n=== Stress Test Complete ===")
    print(f"Run ID: {run_id}")
    print(f"Prompts: {len(prompts)}")
    print(f"Valid outputs: {valid_count}/{len(prompts)}")
    print(f"JSON output folder: {dirs['json']}")
    print(f"Download output folder: {dirs['downloads']}")
    print(f"JSON summary: {summary_json_path}")
    print(f"Analysis report: {summary_md_path}")

    return 0 if valid_count == len(prompts) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stress test generator API with many prompts")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--count", type=int, default=500, help="Generated prompt count")
    parser.add_argument("--max-prompts", type=int, default=None, help="Optional cap after loading/generating prompts")
    parser.add_argument("--prompts-file", default=None, help="Path to prompts file (.txt/.md/.json/.csv)")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrent in-flight prompts")
    parser.add_argument("--poll-interval", type=float, default=1.5, help="Task poll interval seconds")
    parser.add_argument("--task-timeout", type=int, default=300, help="Max seconds waiting per task")
    parser.add_argument("--download-exports", action="store_true", default=True, help="Also call ?download=true endpoint and save bytes")
    parser.add_argument("--no-download-exports", action="store_false", dest="download_exports", help="Skip download=true endpoint")
    parser.add_argument("--no-dedupe", action="store_true", help="Disable prompt deduplication")
    parser.add_argument("--min-prompt-length", type=int, default=10, help="Minimum prompt length after normalization")
    parser.add_argument("--dry-run", action="store_true", help="Only prepare prompts, do not call API")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(execute(parse_args())))
