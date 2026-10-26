# 1. IMPORTS

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import pandas as pd
from vllm import LLM, SamplingParams

# 2. CONFIG

DATA_PATH = "/home/as4638/OMI/data/cath_df_19aug26.csv"
OUTPUT_DIR = "/home/as4638/OMI/data/structured_jsons_full"

TEXT_COLUMN = "CATH_NARRATIVE"
ID_COLUMNS = ["MRN", "ORDER_ID", "ACCESSION_NUM"]

BATCH_SIZE = 32
MAX_ROWS = None

# 4. EXTRACTION INSTRUCTIONS

SYSTEM_PROMPT = """
You are a clinical information extraction model for coronary angiography reports.

Extract ONLY information explicitly stated in the report. Never infer, guess,
or use outside medical knowledge.

Rules:
- study_context: use STEMI, NSTEMI, UA, or stable_elective only when explicitly
  stated. "ACS" alone = unknown.
- Do not classify findings as acute or chronic.
- Do not infer culprit status; only identify a culprit when explicitly stated.
- PCI and stent placement must be explicitly stated.
- TIMI flow: only extract an explicitly stated numeric grade (0-3).
  Never infer TIMI flow from occlusion or qualitative flow descriptions.
- Stenosis: only extract an explicitly stated percentage. A range → min/max;
  a single value → same value for both. Never convert total, complete, or
  subtotal occlusion or qualitative severity into a percentage.
- Keep main vessels separate from branches. pLAD, mid LAD, distal LAD, etc.
  are lesion locations, not branches.
- Represent separately described lesions/vessels.
- Represent bypass grafts separately in graft_findings.
- Missing information = null.
- Never omit required keys.

OMI PROXY LABEL:
- Do not independently infer OMI.
- TIMI 0 or TIMI 1 → OMI.
- Explicit occlusion + TIMI 2 → OMI.
- Explicit severe/total occlusion + TIMI 3 → non-OMI/uncertain.
- Otherwise → non_OMI/uncertain.
- If the required TIMI flow or occlusion information is unavailable → uncertain.

Return ONLY valid JSON. No markdown. No commentary.

Schema:

{
  "study_context": "STEMI/NSTEMI/UA/stable_elective/unknown",

  "acute_culprit_lesion": {
    "present": true/false/null,
    "vessel": "LM/LAD/LCX/RCA/other/null",
    "branch": "string/null",
    "lesion_location": "string/null",
    "acute_occlusion": true/false/null,
    "stenosis_percent_min": number/null,
    "stenosis_percent_max": number/null,
    "timi_flow": 0/1/2/3/null,
    "thrombus_present": true/false/null,
    "thrombus_grade": number/null
  },

  "pci": {
    "performed": true/false/null,
    "culprit_vessel_treated": "string/null",
    "stent_placed": true/false/null
  },

  "other_coronary_findings": [
    {
      "vessel": "LM/LAD/LCX/RCA/other",
      "branch": "string/null",
      "stenosis_percent_min": number/null,
      "stenosis_percent_max": number/null,
      "timi_flow": 0/1/2/3/null
    }
  ],

  "graft_findings": [
    {
      "graft_type": "LIMA/SVG/other",
      "target_vessel": "string/null",
      "status": "patent/occluded/other"
    }
  ],

  "omi_proxy_label": "OMI/non_OMI/uncertain",

  "report_uncertainty": "none/low/moderate/high",
  "uncertainty_reason": "string/null"
}
"""


def format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    minutes, remaining = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)} min {remaining:.2f} sec"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{int(hours)} hr {int(remaining_minutes)} min {remaining:.2f} sec"

# 5. CREATE CHAT MESSAGES

def make_messages(report):

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": f"CATH_NARRATIVE:\n{report}",
        },
    ]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face model name or local model path",
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=MAX_ROWS,
        help="Number of reports to process. Omit to process every row.",
    )

    args = parser.parse_args()

    model_path = args.model
    max_rows = args.max_rows

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=4096,
    )

    # 6. LOAD MODEL

    print(f"Loading model: {model_path}")
    load_start = time.perf_counter()

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        max_model_len=32768,
        gpu_memory_utilization=0.90,
        max_num_seqs=BATCH_SIZE,
        tensor_parallel_size=1,
        trust_remote_code=True,
    )

    load_seconds = time.perf_counter() - load_start
    print(f"Model load time: {format_duration(load_seconds)}")

    # 7. LOAD DATA

    df = pd.read_csv(DATA_PATH)

    if max_rows is not None:
        df = df.head(max_rows)

    print(f"Loaded {len(df)} reports")

    # 8. CREATE OUTPUT DIRECTORY

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 9. RUN INFERENCE

    process_start = time.perf_counter()
    inference_seconds = 0.0

    for start in range(0, len(df), BATCH_SIZE):

        # Get one batch of reports
        batch = df.iloc[start:start + BATCH_SIZE]

        # Create system + user messages for every report
        messages = [
            make_messages(
                "" if pd.isna(report) else str(report)
            )
            for report in batch[TEXT_COLUMN]
        ]

        # vLLM handles:
        #   1. chat template
        #   2. tokenization
        #   3. model generation
        batch_start = time.perf_counter()
        outputs = llm.chat(
            messages,
            sampling_params=sampling_params,
            chat_template_kwargs={
                "enable_thinking": False
            },
        )
        batch_seconds = time.perf_counter() - batch_start
        inference_seconds += batch_seconds

        # 10. PROCESS MODEL OUTPUTS

        for i, output in enumerate(outputs):

            raw_text = output.outputs[0].text

            row = batch.iloc[i]

            # Get identifiers for this report
            ids = {
                column: row[column]
                for column in ID_COLUMNS
                if column in row.index
            }

            # JSON text → Python dictionary
            try:

                extraction = json.loads(raw_text)

                record = {
                    **ids,
                    "extraction": extraction,
                }

                print(
                    f"SUCCESS: "
                    f"{row['ORDER_ID'] if 'ORDER_ID' in row.index else i}"
                )

            except json.JSONDecodeError as error:

                # Model did not return valid JSON
                record = {
                    **ids,
                    "raw_output": raw_text,
                    "parse_error": str(error),
                }

                print(
                    f"FAILED JSON PARSING: "
                    f"{row['ORDER_ID'] if 'ORDER_ID' in row.index else i}"
                )

            # 11. SAVE RESULT

            if "ORDER_ID" in row.index:
                filename = f"{row['ORDER_ID']}.json"
            else:
                filename = f"report_{start + i}.json"

            output_path = output_dir / filename

            output_path.write_text(
                json.dumps(
                    record,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                ),
                encoding="utf-8",
            )

        # 12. PROGRESS

        completed = min(
            start + BATCH_SIZE,
            len(df),
        )

        print(
            f"Completed {completed}/{len(df)} reports "
            f"(batch inference: {format_duration(batch_seconds)})"
        )

    # 13. DONE

    process_seconds = time.perf_counter() - process_start
    n_reports = len(df)
    per_report = (
        inference_seconds / n_reports if n_reports else 0.0
    )

    print("Inference complete.")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Model load time: {format_duration(load_seconds)}")
    print(f"Processing time: {format_duration(process_seconds)}")
    print(f"vLLM generation time: {format_duration(inference_seconds)}")
    if n_reports:
        print(
            f"Average generation time per report: "
            f"{format_duration(per_report)}"
        )


if __name__ == "__main__":
    main()