# Credits

This launcher is a thin layer on top of a tall stack of other people's work.

## Upstream projects

- **[vLLM](https://github.com/vllm-project/vllm)**, the inference engine. Apache-2.0.
- **[devnen/vllm-windows](https://github.com/devnen/vllm-windows)**, the
  patched native-Windows vLLM build that this launcher bundles. Source of
  the three Windows-specific patches (CPU-relay, Qwen3 reasoning parser,
  wildcard model name), see that repo's `CHANGES_VS_SYSTEMPANIC.md`.
- **[SystemPanic/vllm-windows](https://github.com/SystemPanic/vllm-windows)**,
  upstream Windows wheel build infrastructure and the CUDA-12.6 / PyTorch 2.11.0
  base our patched wheel is built against. Without their multi-day MSVC/CUDA
  toolchain wrangling, native vLLM on Windows wouldn't exist at all.
- **[Lorbus](https://huggingface.co/Lorbus)**, produced
  [`Qwen3.6-27B-int4-AutoRound`](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound),
  the only 27B INT4 quant we know of that keeps the MTP head in BF16.
  Without that, MTP spec-decode silently no-ops on 27B and the speed
  numbers in this README don't exist.
- **[Peutlefaire](https://huggingface.co/Peutlefaire)**, produced
  [`Qwen3.6-27B-NVFP4`](https://huggingface.co/Peutlefaire/Qwen3.6-27B-NVFP4),
  the NVFP4 quant with a bundled MTP head that is the Blackwell (RTX
  5090) default since v1.3.0. Routes FFN GEMMs through FlashInfer's
  sm_120 native FP4 tensor cores and escapes the 170 W prefill ceiling
  AutoRound hits on consumer Blackwell.
- **[Qwen team](https://huggingface.co/Qwen)**, Qwen3.6-27B base model.

## Patches we mirror

- **PR [#35687](https://github.com/vllm-project/vllm/pull/35687)**, the
  Qwen3 reasoning parser fix that treats `<tool_call>` as an implicit
  `</think>`. Mirrored verbatim into our `qwen3_reasoning_parser.py`.

## Tools

- **[Textual](https://github.com/Textualize/textual)**, the TUI framework
  the launcher is built on. The portable launcher's footprint is dominated
  by Textual + Rich + their dependencies.
- **[httpx](https://github.com/encode/httpx)**, async HTTP client used by
  the launcher's bench/test path.

## Community signal

The configurations and tradeoffs in this fork were shaped by recipes,
benchmarks, and hard-won failure stories from r/LocalLLaMA, particularly
the people who dared post their numbers and let the community tear them
apart. Specific threads we leaned on are linked from the relevant docs.

If your work is here and you'd like a more direct attribution, please open
an issue.
