# Other models / quants on this launcher

Honest answers about which models you can swap in and which you can't.

The launcher is **tuned end-to-end for `Lorbus/Qwen3.6-27B-int4-AutoRound`**
on Ampere/Ada (3090, 4090, A6000): chat template, tool-call parser,
reasoning parser, MTP head detection, sampler defaults. On Blackwell
(RTX 5090) the default since v1.3.0 is
[`Peutlefaire/Qwen3.6-27B-NVFP4`](https://huggingface.co/Peutlefaire/Qwen3.6-27B-NVFP4)
served by the `rtx5090_nvfp4` snapshot via `--quantization=compressed-tensors`;
AutoRound INT4 stays available as `rtx5090` / `rtx5090_max`.
Swapping the weights for something else may work, but how cleanly
depends on what changes.

## Quick verdict

| Model | Status | Notes |
|---|---|---|
| `Lorbus/Qwen3.6-27B-int4-AutoRound` | ✅ blessed | Default on Ampere/Ada (also a Blackwell alternate). MTP head in BF16, ~17 GB on disk, 64.5 tok/s decode on 3090, 158.1 tok/s on 5090. |
| `Peutlefaire/Qwen3.6-27B-NVFP4` | ✅ blessed (Blackwell default since v1.3.0) | NVFP4 routed via FlashInfer's sm_120 native FP4 tensor cores. Bundled MTP head. `--quantization=compressed-tensors`. Loaded by `rtx5090_nvfp4`. See [`BLACKWELL.md`](BLACKWELL.md), [`SM120_GDN_CEILING.md`](SM120_GDN_CEILING.md). |
| Other Qwen3.6-27B INT4 quants (cyankiwi, groxaxo/Qwen3.6-GPTQ-Pro-4bit, others) | 🟡 boots but no MTP | Loader silently skips the quantised MTP head, draft acceptance ≈ 0%, decode caps at the un-speculated rate (~30-40 tok/s). See [`MTP_HEAD.md`](MTP_HEAD.md). |
| Qwen3.6-27B FP8 | 🟡 untested | Should fit a 24 GB card with smaller ctx. Sampler defaults still apply. Please post numbers. |
| Qwen3.6-27B INT8 | ❌ doesn't fit | Weights ~27 GiB; doesn't fit a 24 GiB card. PP=2 across 2× 24 GB works in principle but means no MTP (PP+MTP broken on 0.19), capping decode near pp2_160k's 43 tok/s. Use INT4 AutoRound. |
| Qwen3 / Qwen3.5-27B (non-thinking variants) | 🟡 mostly works | The reasoning parser expects `<think>` tags; non-thinking models won't emit them, so the reasoning field stays empty. Otherwise serves fine. |
| Qwen3-14B INT4 | 🟢 works on 16 GB cards | Drop-in for 4060 Ti 16G / 4070 Ti / 4080 / 5070. Smaller weights, no MTP head, but vLLM's continuous batching is still a win over llama.cpp. Edit a snapshot to point at the new weights. |
| Qwen3-8B / 4B / 1.7B / 0.6B INT4 | 🟢 works | For 8-12 GB cards. Useful for draft-model spec-decode in theory but vocab mismatch with 27B blocks that path. |
| Qwen3.6-27B "abliterated" / Heretic / Censorship-removed forks | 🟡 works, INT4-AutoRound boots cleanly | See ["Abliterated / heretic / uncensored variants"](#abliterated--heretic--uncensored-variants) below. `lyf/Qwen3.6-27B-heretic-v2-mtp-int4-AutoRound` mirrors Lorbus's recipe (BF16 MTP head preserved). For Blackwell, `sakamakismile/Huihui-Qwen3.6-27B-abliterated-NVFP4-TEXT-MTP`. Other variants boot but may silently no-op MTP — verify via `MTP_HEAD.md`. |
| Llama 3.1, Mistral, Gemma, Phi, etc. | 🟡 boots, wrong defaults | The wheel can serve them, but the shipped chat template, tool-call parser, reasoning parser, and snapshots are Qwen3-specific. You'd need to rebuild the snapshot to swap the template and parsers. Outside what this project does for you. |
| GGUF anything | ❌ won't work | This is vLLM, not llama.cpp. Use the safetensors version of the model. |

## How to swap models

### Three ways to point the launcher at custom weights

Pick whichever fits your workflow. All three trigger the same boot-time
banner — `[model] using <path>  (source: env|--model-dir|saved-config|default|drive-scan)`
— so you can always confirm the launcher picked the right dir.

| Mechanism | When to use | How |
|---|---|---|
| `start.bat --model-dir "<path>"` | One-shot test, scripted launches | `start.bat --model-dir "D:\models\<your-model>"` |
| `VLLM_MODEL_DIR` env var | Multiple snapshots, persists across launches in the same shell | `set VLLM_MODEL_DIR=D:\models\<your-model>` then `start.bat` (PowerShell: `$env:VLLM_MODEL_DIR="..."`) |
| In-TUI model-dir picker (v1.0+) | Permanent default for this install | Launch `start.bat`, the picker shows current model + a Browse button. Selection is saved to `user_config.json` and used on every subsequent boot. |

For the NVFP4 path on Blackwell, the variable is `VLLM_NVFP4_MODEL_DIR`
(separate from `VLLM_MODEL_DIR` so you can keep both an AutoRound and
an NVFP4 model on disk and switch by snapshot).

After any swap, **always**:

1. Patch the tokenizer (idempotent — skips if already patched):
   ```powershell
   python windows_tools\patch_tokenizer.py "D:\models\<your-model>"
   ```
   This flips the `tokenizer_class` from quant-uploader-specific
   classes (e.g. `TokenizersBackend`) that transformers 4.57 doesn't
   recognise to `Qwen2Tokenizer`.
2. Verify shard checksums if you downloaded from HF:
   ```powershell
   python windows_tools\verify_model_sha.py "D:\models\<your-model>"
   ```
3. Boot the snapshot, then run the 3-tier coherence check:
   ```powershell
   python windows_tools\check_coherence.py --port 5001
   ```
4. Watch the boot log for `draft_acceptance_rate`. Near 0.0 means the
   quant's MTP head got silently skipped — see
   [`MTP_HEAD.md`](MTP_HEAD.md) for the safetensors-grep procedure
   that confirms whether `mtp.fc` is in BF16.

### Same model class (Qwen3.6 INT4 AutoRound from a different uploader)

The simplest case — the snapshot's `--quantization=auto-round` flag
and the shipped chat template / tool-call parser / reasoning parser
all apply unchanged. Download into a folder (convention:
`<drive>:\_models\<UploaderName>\<ModelName>\`) and use one of the
three mechanisms above to point the launcher at it. Run the post-swap
steps (tokenizer patch, SHA verify, coherence check, watch
`draft_acceptance_rate`).

### Abliterated / heretic / uncensored variants

These are community fine-tunes of Qwen3.6-27B with the refusal vector
ablated. They serve fine on this launcher as long as you pick a quant
that preserves the MTP head, otherwise speculative decoding silently
no-ops and decode caps near 30-40 tok/s instead of 60+.

**Recommended drop-ins** (verified on HuggingFace, BF16 MTP head
preserved):

| Repo | Use with | Notes |
|---|---|---|
| [`lyf/Qwen3.6-27B-heretic-v2-mtp-int4-AutoRound`](https://huggingface.co/lyf/Qwen3.6-27B-heretic-v2-mtp-int4-AutoRound) | Ampere/Ada zip, any AutoRound INT4 snapshot (`start_speed`, `start_127k`, `start_mtp4`, `rtx5090`, `rtx5090_max`) | INT4 AutoRound, mirrors Lorbus's quantization recipe on a heretic body. `--quantization=auto-round` works as-is. |
| [`sakamakismile/Huihui-Qwen3.6-27B-abliterated-NVFP4-TEXT-MTP`](https://huggingface.co/sakamakismile/Huihui-Qwen3.6-27B-abliterated-NVFP4-TEXT-MTP) | Blackwell zip, `rtx5090_nvfp4` snapshot | NVFP4 with all 15 `mtp.*` tensors in BF16. Use `VLLM_NVFP4_MODEL_DIR` instead of `VLLM_MODEL_DIR`. |

**Other abliterated repos that exist but aren't pre-verified:**
[`hell0ks/Qwen3.6-27B-heretic-ara-int4-AutoRound`](https://huggingface.co/hell0ks/Qwen3.6-27B-heretic-ara-int4-AutoRound),
[`prithivMLmods/Qwen3.6-27B-abliterated-rMAX`](https://huggingface.co/prithivMLmods/Qwen3.6-27B-abliterated-rMAX),
[`wangzhang/Qwen3.6-27B-abliterated`](https://huggingface.co/wangzhang/Qwen3.6-27B-abliterated),
[`acyildirimer/Qwen3.6-27B-int4-AutoRound`](https://huggingface.co/acyildirimer/Qwen3.6-27B-int4-AutoRound).
They boot, but the MTP head BF16 status isn't confirmed — run the
safetensors-grep procedure in [`MTP_HEAD.md`](MTP_HEAD.md) before
trusting any decode tok/s number, or just watch the boot log for
`draft_acceptance_rate`. If it's near 0.0, the quant's MTP head got
silently skipped and you're running un-speculated decode.

**The full-precision base** [`huihui-ai/Huihui-Qwen3.6-27B-abliterated`](https://huggingface.co/huihui-ai/Huihui-Qwen3.6-27B-abliterated)
exists but is 54 GiB on disk — too big for any 24 GB card and too big
even for a 32 GB 5090 with useful KV. Useful only on A100 80 GB or
similar. Quantize it yourself if you want a custom AutoRound INT4
without trusting a third-party uploader; the AutoRound recipe Lorbus
used is in their model card.

**Procedure** is the same as any other Qwen3.6-27B INT4 swap:

```powershell
# 1. Download the weights into your models dir
huggingface-cli download lyf/Qwen3.6-27B-heretic-v2-mtp-int4-AutoRound `
    --local-dir D:\models\Qwen3.6-27B-heretic-v2-mtp-int4-AutoRound

# 2. Patch the tokenizer (idempotent; skips if already patched)
python windows_tools\patch_tokenizer.py D:\models\Qwen3.6-27B-heretic-v2-mtp-int4-AutoRound

# 3. Point the launcher at it (one-shot via CLI flag, or set the env var)
start.bat --model-dir "D:\models\Qwen3.6-27B-heretic-v2-mtp-int4-AutoRound"
```

The launcher prints `[model] using <path>  (source: --model-dir)` at
boot so you can confirm it picked up the swap. Run
[`windows_tools\check_coherence.py --port 5001`](../windows_tools/check_coherence.py)
once it's serving — coherence-validated TPS is the only TPS that
matters, and abliteration occasionally damages the model in ways the
quant can't recover.

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

For other models, download the weights with `huggingface-cli` or
`snapshot_download`, then point the launcher at the local directory
using one of the
["Three ways to point the launcher at custom weights"](#three-ways-to-point-the-launcher-at-custom-weights)
above.

## Related

- [`MTP_HEAD.md`](MTP_HEAD.md), why Lorbus AutoRound specifically and
  how to detect when MTP is silently broken.
- [`HARDWARE.md`](HARDWARE.md), what GPU + VRAM combination fits
  what model size.
- [`COHERENCE.md`](COHERENCE.md), the 3-tier validator. Run it after
  any model swap before trusting a TPS number.
- [`SNAPSHOTS.md`](SNAPSHOTS.md), the in-TUI editor for cloning a
  snapshot to point at a new model.
