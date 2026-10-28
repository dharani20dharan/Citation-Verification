"""Benchmark Runner for Stage 11 Evaluation Benchmark & Error Analysis.

Loads benchmark dataset, executes cases through the Stage 10C pipeline,
evaluates against explicit ground truth, performs stage-level error attribution,
and generates machine-readable JSON and Markdown reports.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluator import BenchmarkEvaluator
from src.evaluation.models import BenchmarkCase, BenchmarkReport, CaseEvaluationResult

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("run_benchmark")


def generate_markdown_report(report: BenchmarkReport) -> str:
    """Generate comprehensive, auditable markdown evaluation report."""
    m = report.metrics
    lines = []
    lines.append("# STAGE 11 — Benchmark Evaluation & Error Analysis Report\n")
    lines.append(f"- **Execution Timestamp**: `{report.timestamp}`")
    lines.append(f"- **Pipeline Version**: `{report.pipeline_version}`")
    lines.append(f"- **Benchmark Version**: `{report.benchmark_version}`\n")

    # 1. Executive Summary
    lines.append("## 1. Executive Summary\n")
    lines.append("| Metric | Value | Description |")
    lines.append("|---|---|---|")
    lines.append(f"| **Total Cases** | `{m.total_cases}` | Total benchmark evaluation cases |")
    lines.append(f"| **Passed Cases** | `{m.passed_cases}` | Cases satisfying all ground truth criteria |")
    lines.append(f"| **Failed Cases** | `{m.failed_cases}` | Cases with facet or evidence divergence |")
    lines.append(f"| **Overall Pass Rate** | `{m.pass_rate * 100:.1f}%` | Percentage of fully compliant cases |")
    lines.append(f"| **Final Verdict Accuracy** | `{m.verdict_accuracy * 100:.1f}%` | Accuracy on final verification verdict |")
    lines.append(f"| **Reference Resolution Accuracy** | `{m.reference_resolution_accuracy * 100:.1f}%` | Exact/not_found discovery accuracy |")
    lines.append(f"| **Source Availability Accuracy** | `{m.source_availability_accuracy * 100:.1f}%` | Attribution & source availability accuracy |")
    lines.append(f"| **Conclusiveness Classification** | `{m.conclusive_classification_accuracy * 100:.1f}%` | Conclusive vs inconclusive accuracy |\n")

    # 2. Stage Metrics
    lines.append("## 2. Stage-Level Performance & Error Attribution\n")
    if m.stage_error_counts:
        lines.append("| Pipeline Stage | Divergence Count | Failure Description |")
        lines.append("|---|---|---|")
        for stage, count in sorted(m.stage_error_counts.items()):
            lines.append(f"| `{stage}` | `{count}` | Earliest divergence detected at stage |")
        lines.append("")
    else:
        lines.append("No stage-level divergences detected across benchmark cases.\n")

    # 3. Verdict Confusion Matrix
    lines.append("## 3. Verdict Confusion Matrix\n")
    lines.append(m.confusion_matrix.to_markdown())
    lines.append("\n")

    # 4. Error Breakdown
    lines.append("## 4. Error Taxonomy Breakdown\n")
    if m.error_taxonomy_counts:
        lines.append("| Error Classification | Count | Description |")
        lines.append("|---|---|---|")
        for err_type, count in sorted(m.error_taxonomy_counts.items()):
            lines.append(f"| `{err_type}` | `{count}` | Classified diagnostic error |")
        lines.append("")
    else:
        lines.append("All cases passed without triggering error taxonomy classifications.\n")

    # 5. Failed Cases
    lines.append("## 5. Failed Cases Detailed Diagnostics\n")
    failed_results = [r for r in report.results if not r.passed]
    if failed_results:
        for r in failed_results:
            diag = r.failure_diagnosis
            lines.append(f"### Case: `{r.case_id}` ({r.category})")
            lines.append(f"- **Expected Verdict**: `{r.expected_verdict}` | **Actual Verdict**: `{r.actual_verdict}`")
            lines.append(f"- **Expected Ref Status**: `{r.expected_reference_status}` | **Actual Ref Status**: `{r.actual_reference_status}`")
            lines.append(f"- **Expected Source**: `{r.expected_source_status}` | **Actual Source**: `{r.actual_source_status}`")
            if diag:
                lines.append(f"- **Earliest Failure Stage**: `{diag.stage.value}`")
                lines.append(f"- **Error Types**: `{', '.join(e.value for e in diag.error_types)}`")
                lines.append(f"- **Diagnostic Reason**: {diag.reason}")
            lines.append(f"- **Evidence Selected**: `{r.evidence_ids}`")
            lines.append(f"- **Verifier Reasoning**: {r.reasoning}\n")
    else:
        lines.append("Zero test cases failed.\n")

    # 6. Evidence Analysis
    lines.append("## 6. Evidence Trace Analysis\n")
    lines.append("| Case ID | Reference | Evidence Considered | Eligible Evidence | Source Attribution | Verdict |")
    lines.append("|---|---|---|---|---|---|")
    for r in report.results:
        ev_str = ", ".join(r.evidence_ids[:2])
        if len(r.evidence_ids) > 2:
            ev_str += f" (+{len(r.evidence_ids) - 2})"
        el_str = ", ".join(r.eligible_evidence_ids[:2])
        if len(r.eligible_evidence_ids) > 2:
            el_str += f" (+{len(r.eligible_evidence_ids) - 2})"
        if not el_str:
            el_str = "(none)"
        lines.append(f"| `{r.case_id}` | `{r.evidence_trace.get('reference', '')}` | `{ev_str}` | `{el_str}` | `{r.actual_source_status}` | `{r.actual_verdict}` |")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 11 benchmark evaluation suite.")
    parser.add_argument(
        "--benchmark-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "evaluation" / "benchmark.jsonl",
        help="Path to benchmark JSONL file (default: data/evaluation/benchmark.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "evaluation",
        help="Output directory for reports (default: evaluation/)",
    )

    args = parser.parse_args()

    evaluator = BenchmarkEvaluator()
    cases = evaluator.load_cases_from_jsonl(args.benchmark_file)

    print("=" * 80)
    print("STAGE 11: EVALUATION BENCHMARK & ERROR ANALYSIS")
    print("=" * 80)
    print(f"Benchmark File   : {args.benchmark_file.name} ({len(cases)} cases)")
    print(f"Pipeline Version : 1.0.0-stage10c")
    print(f"Started at       : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    t0 = time.perf_counter()
    report: BenchmarkReport = evaluator.evaluate_suite(cases)
    elapsed = time.perf_counter() - t0

    m = report.metrics

    print("\n1. BENCHMARK SUMMARY")
    print("-" * 40)
    print(f"Total Cases Evaluated               : {m.total_cases}")
    print(f"Passed Cases                        : {m.passed_cases}")
    print(f"Failed Cases                        : {m.failed_cases}")
    print(f"Overall Pass Rate                   : {m.pass_rate * 100:.1f}%")
    print(f"Final Verdict Accuracy              : {m.verdict_accuracy * 100:.1f}%")
    print(f"Reference Resolution Accuracy       : {m.reference_resolution_accuracy * 100:.1f}%")
    print(f"Source Availability Accuracy        : {m.source_availability_accuracy * 100:.1f}%")
    print(f"Conclusive Classification Accuracy  : {m.conclusive_classification_accuracy * 100:.1f}%")
    print(f"Total Execution Time                : {elapsed:.2f}s")

    print("\n2. VERDICT CONFUSION MATRIX")
    print("-" * 40)
    print(m.confusion_matrix.to_markdown())

    print("\n3. STAGE-LEVEL ERROR BREAKDOWN")
    print("-" * 40)
    if m.stage_error_counts:
        for stage, count in sorted(m.stage_error_counts.items()):
            print(f"  {stage:<24} : {count}")
    else:
        print("  (No stage errors - 100% stage compliance)")

    print("\n4. ERROR TAXONOMY BREAKDOWN")
    print("-" * 40)
    if m.error_taxonomy_counts:
        for err, count in sorted(m.error_taxonomy_counts.items()):
            print(f"  {err:<36} : {count}")
    else:
        print("  (No taxonomy errors detected)")

    print("\n5. INDIVIDUAL CASE TRACE")
    print("-" * 120)
    hdr = f"{'STATUS':<6} | {'CASE ID':<34} | {'REF':<16} | {'REF STAT':<16} | {'SRC STAT':<12} | {'VERDICT':<21} | {'CONF'}"
    print(hdr)
    print("-" * 120)
    for r in report.results:
        tag = "[PASS]" if r.passed else "[FAIL]"
        row = (
            f"{tag:<6} | "
            f"{r.case_id:<34} | "
            f"{r.evidence_trace.get('reference', ''):<16} | "
            f"{r.actual_reference_status:<16} | "
            f"{r.actual_source_status:<12} | "
            f"{r.actual_verdict:<21} | "
            f"{r.confidence:<5.2f}"
        )
        print(row)
    print("-" * 120)

    # Save reports
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "benchmark_results.json"
    md_path = args.output_dir / "benchmark_report.md"

    with open(json_path, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))

    md_content = generate_markdown_report(report)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # Also mirror to data/evaluation/
    data_eval_dir = PROJECT_ROOT / "data" / "evaluation"
    data_eval_dir.mkdir(parents=True, exist_ok=True)
    with open(data_eval_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))
    with open(data_eval_dir / "benchmark_report.md", "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\nSaved machine-readable report to: {json_path}")
    print(f"Saved human-readable report to   : {md_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
