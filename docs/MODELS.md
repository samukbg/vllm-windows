# Other models / quants on this launcher

Honest answers about which models you can swap in and which you can't.

The launcher is **tuned end-to-end for `Lorbus/Qwen3.6-27B-int4-AutoRound`**:
chat template, tool-call parser, reasoning parser, MTP head detection,
sampler defaults. Swapping the weights for something else may work, but
how cleanly depends on what changes.

## Quick verdict

| Model | Status | Notes |
|---|---|---|
| `Lorbus/Qwen3.6-27B-int4-AutoRound` | ✅ blessed | The default. MTP head in BF16, ~17 GB on disk, 64.5 tok/s decode on 3090. |
| Other Qwen3.6-27B INT4 quants (cyankiwi, groxaxo/Qwen3.6-GPTQ-Pro-4bit, others) | 🟡 boots but no MTP | Loader silently skips the quantised MTP head, draft acceptance ≈ 0%, decode caps at the un-speculated rate (~30-40 tok/s). See [`MTP_HEAD.md`](MTP_HEAD.md). |
| Qwen3.6-27B FP8 | 🟡 untested | Should fit a 24 GB card with smaller ctx. Sampler defaults still apply. Please post numbers. |
| Qwen3.6-27B INT8 | ❌ doesn't fit | Weights ~27 GiB; doesn't fit a 24 GiB card. PP=2 across 2× 24 GB works in principle but means no MTP (PP+MTP broken on 0.19), capping decode near pp2_160k's 43 tok/s. Use INT4 AutoRound. |
| Qwen3 / Qwen3.5-27B (non-thinking variants) | 🟡 mostly works | The reasoning parser expects `<think>` tags; non-thinking models won't emit them, so the reasoning field stays empty. Otherwise serves fine. |
| Qwen3-14B INT4 | 🟢 works on 16 GB cards | Drop-in for 4060 Ti 16G / 4070 Ti / 4080 / 5070. Smaller weights, no MTP head, but vLLM's continuous batching is still a win over llama.cpp. Edit a snapshot to point at the new weights. |
| Qwen3-8B / 4B / 1.7B / 0.6B INT4 | 🟢 works | For 8-12 GB cards. Useful for draft-model spec-decode in theory but vocab mismatch with 27B blocks that path. |
| Qwen3.6-27B "abliterated" / Heretic / Censorship-removed forks | 🟡 boots if INT4-AutoRound | Same constraint as any third-party 27B quant: works if the MTP head is in BF16 (rare), boots-without-MTP otherwise. The chat template still applies; reasoning still works. |
| Llama 3.1, Mistral, Gemma, Phi, etc. | 🟡 boots, wrong defaults | The wheel can serve them, but the shipped chat template, tool-call parser, reasoning parser, and snapshots are Qwen3-specific. You'd need to rebuild the snapshot to swap the template and parsers. Outside what this project does for you. |
| GGUF anything | ❌ won't work | This is vLLM, not llama.cpp. Use the safetensors version of the model. |

## How to swap models

### Same model class (Qwen3.6 INT4 AutoRound from a different uploader)

1. Download the new weights into a folder. Convention:
   `<drive>:\_models\<UploaderName>\<ModelName>\`.
2. Set `VLLM_MODEL_DIR` to the new path before launching, or run
   `start.bat --model-dir "<path>"`. The launcher prints
   `[model] using <path>  (source: env|saved-config|default|drive-scan)`
   at boot so you can confirm.
3. Run [`windows_tools\check_coherence.py`](../windows_tools/check_coherence.py)
   first — coherence-validated TPS is the only TPS that matters.
4. If MTP acceptance is near zero, the quant's MTP head got silently
   skipped. See [`MTP_HEAD.md`](MTP_HEAD.md) for the safetensors-grep
   procedure to confirm.

### Different size of Qwen (e.g. Qwen3-14B INT4 on a 16 GB card)

1. Download the new weights.
2. Open the snapshot editor (`e` on the dashboard) and **Duplicate**
   `start_speed` to a new id like `qwen3_14b`.
3. Edit:
   - **GPU**: `GPU0` if single-GPU
   - **Context**: 64000 to start (14B fits more KV than 27B at the
     same VRAM)
   - **MTP n**: blank (Qwen3-14B has no MTP head)
   - **mem_util**: 0.92 if display-attached, 0.948 if headless
4. Save. The launcher rewrites both `configs.yaml` and the new
   `start_qwen3_14b.py` for you.
5. Set `VLLM_MODEL_DIR` to the 14B weights path and launch the new
   snapshot.

### Different model family (Llama, Mistral, Gemma)

This is no longer a "swap weights" operation — the chat template,
tool-call parser, and reasoning parser baked into every snapshot are
Qwen3-specific. You'd need to:

1. Find the right chat template for your target model and drop it
   into `templates\`.
2. Hand-write a new snapshot `.py` that uses
   `--chat-template=<your-template>.jinja`, drops the
   `--tool-call-parser=qwen3_coder` and `--reasoning-parser=qwen3`
   flags, and sets the right `--quantization` for whatever quant
   format the model ships in.
3. Re-run the [3-tier coherence check](COHERENCE.md) and re-bench.

The bundled wheel itself supports any model vLLM 0.19/0.20 supports;
the launcher just isn't packaged for that case. If you do this and
get something working, a PR with the new snapshot is welcome.

## Why "INT4 AutoRound" specifically

INT4 AutoRound is the sweet spot on 24 GB cards:

- KLD vs INT8 on Qwen3.6 is small (a few hundredths of a bit per
  token in test prompts), so output quality is essentially
  indistinguishable from FP8.
- 16.96 GiB on disk leaves enough VRAM for 90-127 k context with MTP
  on a 24 GB card.
- Marlin INT4 kernels on Ampere/Ada/Blackwell are the fastest path
  vLLM has for 4-bit weights on consumer NVIDIA.
- AutoRound preserves the MTP head in BF16, which keeps spec-decode
  working — almost no other 4-bit quantizer does this for
  Qwen3.6-27B.

If you have a 32+ GiB card (5090, A6000, A100) and want to try INT8
or even FP8 with full ctx, please run `check_coherence.py` and post
the numbers. Configs welcome.

## Why the launcher won't auto-download non-default models

The auto-download path in `launcher\app\model_setup.py` is hardcoded
to `Lorbus/Qwen3.6-27B-int4-AutoRound` because that's the model the
shipped snapshots are validated against. If we let it auto-download
arbitrary HuggingFace ids, every "TPS is bad" issue would come with a
"...with this 4-bit GPTQ quant from someone you've never heard of"
caveat we can't reasonably validate. The ergonomics aren't worth the
support cost.

For other models, point the launcher at an existing local directory
(`VLLM_MODEL_DIR` env var, `--model-dir` CLI flag, or the in-TUI
model-picker), and download with `huggingface-cli` or
`snapshot_download` separately.

## Related

- [`MTP_HEAD.md`](MTP_HEAD.md), why Lorbus AutoRound specifically and
  how to detect when MTP is silently broken.
- [`HARDWARE.md`](HARDWARE.md), what GPU + VRAM combination fits
  what model size.
- [`COHERENCE.md`](COHERENCE.md), the 3-tier validator. Run it after
  any model swap before trusting a TPS number.
- [`SNAPSHOTS.md`](SNAPSHOTS.md), the in-TUI editor for cloning a
  snapshot to point at a new model.
