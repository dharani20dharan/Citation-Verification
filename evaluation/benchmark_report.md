# STAGE 11 — Benchmark Evaluation & Error Analysis Report

- **Execution Timestamp**: `2026-09-29 10:57:51`
- **Pipeline Version**: `1.0.0-stage10c`
- **Benchmark Version**: `1.0.0-stage11`

## 1. Executive Summary

| Metric | Value | Description |
|---|---|---|
| **Total Cases** | `16` | Total benchmark evaluation cases |
| **Passed Cases** | `16` | Cases satisfying all ground truth criteria |
| **Failed Cases** | `0` | Cases with facet or evidence divergence |
| **Overall Pass Rate** | `100.0%` | Percentage of fully compliant cases |
| **Final Verdict Accuracy** | `100.0%` | Accuracy on final verification verdict |
| **Reference Resolution Accuracy** | `100.0%` | Exact/not_found discovery accuracy |
| **Source Availability Accuracy** | `100.0%` | Attribution & source availability accuracy |
| **Conclusiveness Classification** | `100.0%` | Conclusive vs inconclusive accuracy |

## 2. Stage-Level Performance & Error Attribution

No stage-level divergences detected across benchmark cases.

## 3. Verdict Confusion Matrix

| Expected \ Actual | supported | unsupported | partially_correct | not_found | insufficient_evidence | wrong_reference | Total |
|---|---|---|---|---|---|---|---|
| **supported** | 3 | 0 | 0 | 0 | 0 | 0 | 3 |
| **unsupported** | 0 | 2 | 0 | 0 | 0 | 0 | 2 |
| **partially_correct** | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| **not_found** | 0 | 0 | 0 | 4 | 0 | 0 | 4 |
| **insufficient_evidence** | 0 | 0 | 0 | 0 | 6 | 0 | 6 |
| **wrong_reference** | 0 | 0 | 0 | 0 | 0 | 0 | 0 |


## 4. Error Taxonomy Breakdown

All cases passed without triggering error taxonomy classifications.

## 5. Failed Cases Detailed Diagnostics

Zero test cases failed.

## 6. Evidence Trace Analysis

| Case ID | Reference | Evidence Considered | Eligible Evidence | Source Attribution | Verdict |
|---|---|---|---|---|---|
| `bench_legal_01_sec_3d_exact` | `Section 3(d)` | `legal_sample_fixture_sec_3_d_chunk_001, legal_sample_fixture_sec_3_d_chunk_002 (+3)` | `legal_sample_fixture_sec_3_d_chunk_001, legal_sample_fixture_sec_3_d_chunk_002 (+1)` | `available` | `supported` |
| `bench_legal_02_sec_9_missing` | `Section 9` | `legal_sample_fixture_sec_3_d_chunk_002, legal_sample_fixture_sec_3_d_chunk_003 (+3)` | `legal_sample_fixture_sec_3_d_chunk_002, legal_sample_fixture_sec_3_d_chunk_003 (+1)` | `available` | `not_found` |
| `bench_legal_03_sec_3e_wrong_ref` | `Section 3(e)` | `legal_sample_fixture_sec_3_d_chunk_002, legal_sample_fixture_sec_3_d_chunk_001 (+3)` | `legal_sample_fixture_sec_3_d_chunk_002, legal_sample_fixture_sec_3_d_chunk_001 (+1)` | `available` | `not_found` |
| `bench_legal_04_sec_3di_nested` | `Section 3(d)(i)` | `legal_sample_fixture_sec_3_d_chunk_003, legal_sample_fixture_sec_3_d_chunk_002 (+3)` | `legal_sample_fixture_sec_3_d_chunk_003, legal_sample_fixture_sec_3_d_chunk_002 (+1)` | `available` | `not_found` |
| `bench_legal_05_heading_rejection` | `Section 3(d)` | `legal_sample_fixture_sec_3_d_chunk_001, legal_sample_fixture_sec_3_d_chunk_002 (+3)` | `legal_sample_fixture_sec_3_d_chunk_001, legal_sample_fixture_sec_3_d_chunk_002 (+1)` | `available` | `unsupported` |
| `bench_sci_01_grouped_3_4` | `[3, 4]` | `scientific_sample_fixture_sec_003_2_methodology_table_003, scientific_sample_fixture_sec_003_2_methodology_chunk_002 (+3)` | `(none)` | `unavailable` | `insufficient_evidence` |
| `bench_sci_02_range_5_7` | `[5–7]` | `scientific_sample_fixture_sec_003_2_methodology_table_003, scientific_sample_fixture_sec_003_2_methodology_chunk_002 (+3)` | `(none)` | `unavailable` | `insufficient_evidence` |
| `bench_sci_03_author_year_supported` | `(Smith et al., 2024)` | `sci_doc_smith_2024_chunk_001, scientific_sample_fixture_sec_003_2_methodology_table_003 (+3)` | `sci_doc_smith_2024_chunk_001` | `available` | `supported` |
| `bench_sci_04_source_unavail_12` | `[12]` | `scientific_sample_fixture_sec_002_1_introduction_chunk_001, scientific_sample_fixture_sec_003_2_methodology_table_003 (+3)` | `(none)` | `unavailable` | `insufficient_evidence` |
| `bench_sci_05_unrelated_source_isolation` | `[12]` | `scientific_sample_fixture_sec_003_2_methodology_chunk_002, scientific_sample_fixture_sec_003_2_methodology_table_003 (+3)` | `(none)` | `unavailable` | `insufficient_evidence` |
| `bench_sci_06_citing_doc_isolation` | `[12]` | `scientific_sample_fixture_sec_002_1_introduction_chunk_001, scientific_sample_fixture_sec_003_2_methodology_table_003 (+3)` | `(none)` | `unavailable` | `insufficient_evidence` |
| `bench_sci_07_contradiction` | `(Smith et al., 2024)` | `sci_doc_smith_2024_chunk_001, scientific_sample_fixture_sec_003_2_methodology_table_003 (+3)` | `sci_doc_smith_2024_chunk_001` | `available` | `unsupported` |
| `bench_sci_08_partial_support` | `(Smith et al., 2024)` | `sci_doc_smith_2024_chunk_001, scientific_sample_fixture_sec_003_2_methodology_table_003 (+3)` | `sci_doc_smith_2024_chunk_001` | `available` | `partially_correct` |
| `bench_sci_09_insufficient_evidence` | `(Smith et al., 2024)` | `sci_doc_smith_2024_chunk_001, scientific_sample_fixture_sec_003_2_methodology_table_003 (+3)` | `sci_doc_smith_2024_chunk_001` | `available` | `insufficient_evidence` |
| `bench_sci_10_missing_reference_guard` | `[999]` | `scientific_sample_fixture_sec_003_2_methodology_table_003, scientific_sample_fixture_sec_003_2_methodology_chunk_002 (+3)` | `(none)` | `unresolved_reference` | `not_found` |
| `bench_legal_07_substantive_support` | `Section 3(d)` | `legal_sample_fixture_sec_3_d_chunk_001, legal_sample_fixture_sec_3_d_chunk_002 (+3)` | `legal_sample_fixture_sec_3_d_chunk_001, legal_sample_fixture_sec_3_d_chunk_002 (+1)` | `available` | `supported` |
