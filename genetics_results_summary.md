# TTR genetic testing from clinical notes: preliminary results (Nov 10, 2026)

**Population:** 634 corroborated ATTR-CM patients.
**Method:** a single-task LLM (Qwen2.5-7B) read every note separately. The answers were cleaned in code (non-TTR genes, impossible variant positions, off-topic quotes and statements about relatives) and rolled up to one row per patient. The extraction code is `equa_attr_genetics_v3.py` and the roll-up is `genetics_v3_rollup_v3.py`; see `genetics_results_634.png`.

**Status:** preliminary. Validation against curated data and a held-out note review are pending.

## Patient-level TTR genetic status (n = 634)

| Status | Main | Verbatim quotes only | TTR-named quotes only |
|---|---|---|---|
| Positive | 43 | 41 | 44 |
| VUS | 1 | 1 | 1 |
| Negative (wild-type) | 153 | 128 | 144 |
| Conflicting | 5 | 5 | 4 |
| Tested, no result | 33 | 47 | 26 |
| Not tested | 119 | 127 | 80 |
| Not documented | 280 | 285 | 335 |
| **Result documented** | **202 (31.9%)** | 175 (27.6%) | 193 (30.4%) |

- **Main:** all cleaned answers.
- **Verbatim quotes only:** only answers whose evidence quote appears word for word in the note (checks that the model did not reword or invent its evidence).
- **TTR-named quotes only:** only answers whose quote names TTR, ATTR, amyloid, wild-type or a TTR variant (checks that results are TTR-specific).

**Key points:**
- The result is stable across the three runs.
- A variant is named for 46 of 48 positive patients (positive or conflicting):
  - V142I (V122I): 41;
  - V50M, T80A, F64S, D58A: 1 each;
  - 1 other spelling;
  - 2 with no variant named.
- **Documented result by age group** (age at last follow-up):
  - <65: 74%;
  - 65–74: 54%;
  - 75–84: 34%;
  - ≥85: 23%.
- **Not tested** (main run): not done 53, planned or recommended 44, unclear 14, declined 8.

The raw roll-up logs (counts only) are kept on the analysis server.
