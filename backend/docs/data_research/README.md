# Data Research Report

This is the detailed, human-readable companion to [`backend/notebooks/DATA_CONTEXT.md`](../../notebooks/DATA_CONTEXT.md). `DATA_CONTEXT.md` is deliberately terse — a quick-reference table meant to be skimmed by a notebook or by Claude before touching the data. This directory is the opposite: full paragraphs, explaining not just *what* the facts are but *why* they're true and what they imply, written so that reading through it end to end (or dipping into just one file) builds genuine, durable understanding of this dataset — not just a lookup table you have to re-derive intuition from every time.

Every concrete number in these files (effect sizes, file counts, accuracy figures) was verified against the executed EDA notebooks, `DATA_CONTEXT.md`, or `backend/dev/change_8_2026-08-16.md` at the time of writing — nothing here is invented or estimated.

## Files, in suggested reading order

**[`01_origin_and_protocol.md`](01_origin_and_protocol.md)** — Start here if you're new to this dataset. Explains where MODMA comes from, exactly who was recruited into the MDD and HC groups and under what clinical criteria (including the important asymmetry between the two groups' screening rigor), the recording equipment and environment, and the full 29-file session protocol — what each task type is, and the clinical/psychological reasoning behind why it was designed that way.

**[`02_metadata_schema_deep_dive.md`](02_metadata_schema_deep_dive.md)** — A paragraph on every column in the metadata workbook: the identity/demographic columns, and a full clinical explanation of each of the six scales (PHQ-9, GAD-7, PSQI, CTQ-SF, LES, SSRS) — what each measures, how it's scored, and, critically, which one (PHQ-9) is baked into the class label itself and therefore unusable as a model feature.

**[`03_label_terminology_history.md`](03_label_terminology_history.md)** — A direct, complete answer to "what's the difference between MDD/HC and MDD/NC": where "NC" genuinely came from in the source paper, how it leaked into this repo's original code, exactly which files still say `NC` today, and the corrected present-day truth (`MDD`/`HC`, verified against all 52 subjects).

**[`04_eda_findings_and_statistics.md`](04_eda_findings_and_statistics.md)** — A narrative recap of everything our own four rebuilt EDA notebooks actually found: the 5 corrupt files, the metadata separation statistics (GAD-7 and PSQI as the strongest legitimate signals), the audio-level/silence-detection finding, and the acoustic feature separability results — tied together into one story instead of scattered across four notebooks.

**[`05_companion_paper_methodology.md`](05_companion_paper_methodology.md)** — Prior art: how an earlier study on this exact 52-subject cohort engineered its features (a much richer 1609-dimensional set), selected among them, and fused 29 per-file predictions into one subject-level verdict to reach 75.8%/68.5% accuracy — useful groundwork for any future modeling notebook in this repository, including the paper's own honest caveats about its results' generalizability.

**[`06_license_and_compliance.md`](06_license_and_compliance.md)** — What the dataset's license actually permits (academic use only, no redistribution beyond a narrow publication exception, a citation requirement) and a real, currently-unresolved compliance gap: no per-subject publication-consent flag exists in the metadata as documented, which matters before sharing any subject-identifiable derived material externally.

## How this relates to the rest of the project's documentation

`CLAUDE.md` (repo root) and `DATA_CONTEXT.md` both link here. Think of the three layers as: `CLAUDE.md` tells you *that* you need to know about the data and where to look; `DATA_CONTEXT.md` gives you the facts fast, for use while actively writing code or notebooks; this directory gives you the understanding behind those facts, for use when you want to actually think clearly about the dataset — designing a new analysis, writing up results, or just refreshing your own expertise before a conversation about the project.
