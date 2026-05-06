# Agent install prompt

Copy/paste prompt for handing the install off to a coding agent (Claude
Code, Cursor, Codex CLI, etc.) on a Windows machine with a CUDA-capable
NVIDIA GPU. The agent does the entire install hands-off and reports
back when inference works.

Edit the `INSTALL_DIR` line before sending, everything else is generic.

---

```text
Install qwen3.6-windows-server (latest release) end-to-end on this Windows
machine and verify it serves inference. This is hands-off, complete every
step yourself, do not ask me to drive a TUI.

INSTALL_DIR: C:\path\to\install        # <-- EDIT THIS LINE before sending

Source of truth: https://github.com/devnen/qwen3.6-windows-server/releases/latest

Hardware assumed: NVIDIA Ampere or newer (RTX 3090 / 4090 / 5090 / A6000),
recent CUDA driver. Won't work on Pascal/Turing/Intel Arc/AMD.

## Steps

1. Disk check. Ensure >= 40 GB free on the INSTALL_DIR drive (model is
   ~16 GB, runtime is ~5 GB, plus pip cache + temp).

2. Detect the GPU arch so you pick the right zip. Two zips ship per
   release:

   - `qwen3.6-windows-server-portable-x64.zip` for Ampere / Ada
     (RTX 3090 / 4090 / A6000, sm_86 / sm_89).
   - `qwen3.6-windows-server-portable-x64-blackwell.zip` for Blackwell
     (RTX 5060 / 5070 / 5080 / 5090, sm_120). The default zip lacks
     sm_120 kernels and dies at boot with
     `cudaErrorNoKernelImageForDevice` on a 50-series card.

   Quick way to choose:

      nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1

   `12.0` -> Blackwell zip. `8.6` / `8.9` -> default zip.

3. Create INSTALL_DIR if missing. Use the GitHub API to find the latest
   release zip URL for the matching variant, don't hardcode a tag, the
   project ships fixes regularly:

      # default (Ampere/Ada) zip
      curl -sL https://api.github.com/repos/devnen/qwen3.6-windows-server/releases/latest \
        | grep -oE '"browser_download_url": *"[^"]*portable-x64\.zip"' \
        | head -1 | cut -d'"' -f4

      # Blackwell zip
      curl -sL https://api.github.com/repos/devnen/qwen3.6-windows-server/releases/latest \
        | grep -oE '"browser_download_url": *"[^"]*portable-x64-blackwell\.zip"' \
        | head -1 | cut -d'"' -f4

   Download that zip into INSTALL_DIR, then `unzip -q` it. Result: a
   `qwen3.6-windows-server\` subfolder containing `start.bat`,
   `python\`, `wheels\`, `launcher\`, etc.

4. Pick a snapshot to launch. Snapshot ids depend on the zip variant:

   - **Default (Ampere/Ada) zip:** `start_72tps` for short-prompt
     headless 3090/4090/A6000, `start_gpu0_50k` if the card also drives
     a display.
   - **Blackwell zip:** `rtx5090` for the default 240k-context profile,
     `rtx5090_max` for the 280k variant. These work on any sm_120
     card; on a 5060 / 5070 / 5080 the smaller VRAM may force you to
     drop context, but the snapshot ids are the same.

5. Launch headlessly. The launcher is a Textual TUI by default but has
   full CLI flags. From bash, you must invoke the .bat through
   `cmd.exe` with an absolute path, relative paths and bare `start.bat`
   don't resolve across the bash->cmd boundary. Substitute the snapshot
   id you picked in step 4:

      cmd.exe //c 'C:\absolute\path\to\qwen3.6-windows-server\start.bat \
        --auto-download --snapshot <SNAPSHOT_ID> --yes' \
        > "$INSTALL_DIR/launcher.log" 2>&1 &

   Flag notes:
   - `--auto-download` is safe to always pass: the launcher first scans
     fixed drives for an existing `Qwen3.6-27B-int4-AutoRound` directory
     (under `<drive>:\`, `_models\`, `models\`, `AI\`, `huggingface\hub\`,
     `models\Lorbus\`, etc.) and only downloads if none is found.
   - As of v0.1.7 the launcher prints
     `[model] using <path>  (source: env|saved-config|default|drive-scan)`
     at boot. If it picks a non-Lorbus AutoRound dir, it warns. To force
     a specific directory: replace `--auto-download` with
     `--model-dir "X:\path\to\weights"`.
   - `start.bat` already detects `MSYSTEM` / `TERM` / `CI` and stays
     in-place rather than detaching into Windows Terminal, so the bash
     environment is fine. (`VLLM_NO_WT=1` is belt-and-suspenders if you
     want it.)
   - DO NOT trust the bash background-task "completed" signal, cmd.exe
     returns once the script chain detaches, but the python.exe children
     keep installing and serving for several minutes after. Poll the log
     and the HTTP endpoint, not the spawn handle.

4. Monitor progress. First run does two slow stages, both visible in
   `$INSTALL_DIR/launcher.log`:
   - Runtime install: ~5-15 min (vLLM + torch + ~150 deps, ~3.5 GB
     downloaded). Ends with `[setup] vLLM runtime installed.`
   - vLLM cold boot: ~2 min. Look for the snapshot banner
     (`vLLM serve: qwen3.6-27b-autoround`) followed eventually by
     `Application startup complete.`
   The vLLM serving process additionally tees its own stdout to
   `$INSTALL_DIR/qwen3.6-windows-server/logs/vllm_server.5001.log`,
   tail that for engine-side progress.

5. Wait for the server to actually accept requests. Poll, don't guess:

      for i in $(seq 1 90); do
        curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:5001/v1/models \
          | grep -q 200 && break
        sleep 5
      done

   Cap the wait at ~10 min total. If it never returns 200, dump the
   tail of both log files for diagnosis.

6. Smoke test inference:

      curl -s -X POST http://127.0.0.1:5001/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"any","messages":[{"role":"user","content":"Capital of France?"}],"max_tokens":2000}' \
        > "$INSTALL_DIR/response.json"

   Parse the response. The answer "Paris" lands in
   `choices[0].message.content`. Chain-of-thought lands in
   `choices[0].message.reasoning` (that's the `--reasoning-parser=qwen3`
   patch, not a bug). `max_tokens: 2000` matters, Qwen3.6 is a thinking
   model and `max_tokens: 50` will be eaten by the thinking phase, leaving
   `content: null` with `finish_reason: "length"`.

## Success criteria (all three must hold before reporting done)

- HTTP GET `http://127.0.0.1:5001/v1/models` returns 200.
- The smoke-test POST returns "Paris" in `content` (or in `reasoning`
  if the wheel happens to leave it there, accept either).
- The launcher log shows a single `[model] using <path>  (source: ...)`
  line and you can report which directory was used.

## On failure

- Read `docs/TROUBLESHOOTING.md` inside the extracted folder before
  improvising, most failure modes (KV cache OOM, wrong attention
  backend, port-in-use, tokenizer class mismatch) are pre-diagnosed
  there with exact fixes.
- Do not try to "fix" things by editing files inside the extracted
  release. Re-download is faster and produces a known state.
- For single-GPU hosts on the default zip where GPU 0 has the desktop
  attached, swap `--snapshot start_72tps` for `--snapshot start_gpu0_50k`.
- On the Blackwell zip both `rtx5090` and `rtx5090_max` already use
  `mem_util=0.95` and tolerate a typical desktop tax on a 32 GB 5090.
  If a smaller Blackwell card OOMs at boot, lower `--max-model-len` in
  the snapshot before retrying (vLLM prints the safe ceiling in the
  error).

Report back when all three success criteria hold, with: the tag that was
installed, the model directory the launcher picked (from the
`[model] using` line), and the first decoded answer.
```
