# Release validation — 2026-10-15

| Check | Outcome |
|---|---|
| Original files downloaded and hashed | 481 snapshot files matched their server-side SHA-256; four later launch assets matched recorded launch-manifest hashes |
| Published source/result provenance | 327 copied files checked; explicit description/privacy edits recorded |
| Python AST parsing | 62 files passed |
| Existing test suite | **79 passed** (`pytest`, 6.75 seconds on local CPU) |
| Paper YAML parser validation | 18 configs parsed successfully using their actual entry points |
| Full-model smoke check | 12,149,928 parameters; input `(1,12,2500)` + `(1,7)` → output `(1,12)` |
| Frozen results | 25 groups have exactly seeds `{42,2026,3407}`; means and sample SDs recomputed from exact seed CSVs |
| Manuscript source references | 32 input/figure references resolved, including declared `graphicspath` |
| Later attribution version | All three existing attribution tests passed after replacing the older script with the actual three-seed launch version |
| Ensemble statistics smoke | 80 synthetic records, 20 bootstrap draws; identical model predictions yielded exactly zero paired point differences and intervals |

Local verification used Python 3.12.3 and PyTorch 2.3.1. These checks cover imports, numerical metrics, architecture, configuration parsing and lightweight CPU behavior; they do not replace the historical training environment. The reproduction environment is documented separately from these verification versions.

No training, checkpoint regeneration, patient-level result recomputation or new manuscript PDF compilation was performed. Existing results were preserved byte-for-byte. The public result inventory excludes patient-level arrays and the original training weights.

Initial verification correctly prompted two packaging corrections: the manuscript resolver was extended to honor `graphicspath`; continuation-only YAML seed lists were expanded to the full paper seed set. The low-label example uses the recorded permutation seed `20260801`. These corrections change the release wrapper/documentation, not recorded experimental results.
