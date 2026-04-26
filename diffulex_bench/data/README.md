# Diffulex JSON benchmarks

Bundled tasks (lm-eval names, use with `--include_path diffulex_bench/tasks` — the bench adds this by default):

| JSON | Task name |
|------|-----------|
| `GSM8K.json` | `gsm8k_diffulex` |
| `MATH500.json` | `math500_diffulex`, `math500_diffulex_4shot`, `math500_diffulex_n32` |
| `HumanEval.json` | `humaneval_diffulex` |
| `MBPP.json` | `mbpp_diffulex` |
| `HumanEval_dmax.json` | `humaneval_dmax_reference_chat` |
| `MBPP_dmax.json` | `mbpp_dmax_reference_chat` |

The `_dmax` JSON files are dedicated copies for DMax-specific code tasks so their prompt/output
contract can evolve independently from the generic `diffulex` / SDAR-style tasks.
