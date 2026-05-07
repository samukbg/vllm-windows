# qwen3.6-windows-server v1.3.3

Bug-fix release. Unblocks pipeline-parallel (PP=2) on the Ampere wheel,
fixes the embedded `bench_summarize.py` import path, and ships a real
~25 k-token long-prompt fixture so documented `decode_tps` numbers are
reproducible from a stock install.

## What changed

- **New patched wheels: `vllm-0.19.0+devnen.2` (Ampere) and
  `vllm-0.20.0+cu132.devnen.2` (Blackwell).** Both add a Windows-only
  ZMQ `ipc://` -> `tcp://` fallback so `get_open_zmq_ipc_path()` works
  on Windows (pyzmq has no `ipc://` transport). The Ampere wheel
  additionally widens a worker-pipe `isinstance` check to
  `_ConnectionBase` so PP=2 boots past the `wait_for_ready` assert. See
  [devnen/vllm-windows v0.19.0-devnen.2](https://github.com/devnen/vllm-windows/releases/tag/v0.19.0-devnen.2)
  for the diff.
- **`pp2_160k` snapshot is functional again on the public Ampere
  release.** Verified on a 2× RTX 3090 box (Designare) — boots, is
  coherent, decodes within ~10 % of the documented 40.3 tok/s. Prior
  releases shipped the wheel without those Windows fixups, so any user
  clicking the "Both-GPU big-ctx" card hit a `ZMQError: Protocol not
  supported` immediately.
- **`bench_summarize.py` now runs from a stock install** without a
  wrapper. `windows_tools/build_launcher_zip.py` adds `..\windows_tools`
  to the embedded `python312._pth`, so `bench_summarize.py`'s
  `import bench` (its sibling module) resolves correctly. Embedded
  Python ignores `cwd` and `PYTHONPATH`, so the `_pth` line was the only
  fix.
- **`bench_prompt_sample.py` is now a real ~130 KB / ~25 k-token
  fixture** (verbatim copy of CPython 3.12's `Lib/inspect.py` under the
  PSF Agreement). Replaces the 670-token stub. Documented `decode_tps`
  numbers are now reproducible from a clean install instead of being
  artificially fast on a placeholder prompt.

## Who is affected

- 2-GPU users who clicked `pp2_160k` on the launcher dashboard — those
  setups are usable again.
- Anyone running `windows_tools\bench_summarize.py` against the
  documented `decode_tps` figures — the regime now matches.
- Single-GPU users see no functional change; the new wheel is a strict
  superset of the +devnen.1 patch set.

## Upgrading

`update.bat` — automatic from v1.3.2. The Ampere variant pulls the new
+devnen.2 wheel and re-runs `setup.ensure_runtime()`. Blackwell
variant likewise.

## Verification

After upgrading, a quick smoke test:

```
cd C:\<install>\windows_tools
..\python\python.exe bench_summarize.py --label v1.3.3-test
```

Expect TSV-appended bench output in `windows_tools/runs.tsv`, no
`ModuleNotFoundError`, and a long-prompt prefill / decode comparable
to (or exceeding) the snapshot's documented numbers.
