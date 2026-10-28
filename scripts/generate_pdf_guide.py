"""Script to generate a publication-grade PDF operations & migration guide for Citation Verification."""

from pathlib import Path
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print 'Page X of Y'."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        self.saveState()
        self.setFont("Helvetica-Bold", 7.5)
        self.setFillColor(colors.HexColor("#718096"))

        # Running header (pages 2+)
        if self._pageNumber > 1:
            self.drawString(54, 11 * inch - 36, "AUTOMATED CITATION VERIFICATION  |  OPERATIONS & DATASET MIGRATION GUIDE")
            self.drawRightString(8.5 * inch - 54, 11 * inch - 36, "STAGE 11 AUDITED")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.75)
            self.line(54, 11 * inch - 42, 8.5 * inch - 54, 11 * inch - 42)

        # Running footer (all pages)
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#718096"))
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(8.5 * inch - 54, 32, page_str)
        self.drawString(54, 32, "CONFIDENTIAL & PROPRIETARY  --  CAPSTONE VERIFICATION ENGINE")
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.75)
        self.line(54, 42, 8.5 * inch - 54, 42)

        self.restoreState()


def create_callout(title: str, text: str, styles, border_color="#3182CE", bg_color="#EBF8FF"):
    """Create a high-visibility styled callout box."""
    title_p = Paragraph(f"<b><font color='{border_color}'>{title}</font></b>", styles["CalloutTitle"])
    content_p = Paragraph(text, styles["CalloutText"])
    callout_data = [[title_p], [content_p]]
    callout_table = Table(callout_data, colWidths=[7.0 * inch])
    callout_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(bg_color)),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LINEBEFORE', (0,0), (0,-1), 3.5, colors.HexColor(border_color)),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
    ]))
    return callout_table


def create_code_block(code_text: str, styles, title: str = None):
    """Create a formatted code snippet block."""
    items = []
    if title:
        items.append([Paragraph(f"<b>{title}</b>", styles["CodeHeader"])])
    
    code_html = code_text.strip().replace("\n", "<br/>").replace(" ", "&nbsp;")
    items.append([Paragraph(f"<font face='Courier' color='#1A202C'>{code_html}</font>", styles["CodeBody"])])
    
    t = Table(items, colWidths=[7.0 * inch])
    bg_style = [
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
    ]
    if title:
        bg_style.extend([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#EDF2F7")),
            ('LINEBELOW', (0,0), (-1,0), 0.5, colors.HexColor("#CBD5E0")),
            ('BOTTOMPADDING', (0,0), (-1,0), 4),
            ('TOPPADDING', (0,0), (-1,0), 4),
        ])
    t.setStyle(TableStyle(bg_style))
    return t


def build_pdf(output_path: Path):
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=52,
        bottomMargin=52
    )

    base_styles = getSampleStyleSheet()

    styles = {}
    styles["DocTitle"] = ParagraphStyle(
        "DocTitle",
        parent=base_styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=21,
        leading=25,
        textColor=colors.HexColor("#1A365D"),
        alignment=TA_LEFT,
        spaceAfter=4
    )

    styles["DocSubtitle"] = ParagraphStyle(
        "DocSubtitle",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#4A5568"),
        alignment=TA_LEFT,
        spaceAfter=10
    )

    styles["SectionHeading"] = ParagraphStyle(
        "SectionHeading",
        parent=base_styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    styles["SubSectionHeading"] = ParagraphStyle(
        "SubSectionHeading",
        parent=base_styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#2D3748"),
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True
    )

    styles["Body"] = ParagraphStyle(
        "Body",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=4,
        alignment=TA_JUSTIFY
    )

    styles["Bullet"] = ParagraphStyle(
        "Bullet",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#2D3748"),
        leftIndent=14,
        firstLineIndent=-9,
        spaceAfter=3
    )

    styles["CalloutTitle"] = ParagraphStyle(
        "CalloutTitle",
        parent=base_styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        spaceAfter=2
    )

    styles["CalloutText"] = ParagraphStyle(
        "CalloutText",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11.5,
        textColor=colors.HexColor("#2D3748")
    )

    styles["CodeHeader"] = ParagraphStyle(
        "CodeHeader",
        parent=base_styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#2D3748")
    )

    styles["CodeBody"] = ParagraphStyle(
        "CodeBody",
        parent=base_styles["Normal"],
        fontName="Courier",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#1A202C")
    )

    styles["TableHeader"] = ParagraphStyle(
        "TableHeader",
        parent=base_styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10.5,
        textColor=colors.white,
        alignment=TA_LEFT
    )

    styles["TableCell"] = ParagraphStyle(
        "TableCell",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#2D3748"),
        alignment=TA_LEFT
    )

    styles["TableCellBold"] = ParagraphStyle(
        "TableCellBold",
        parent=base_styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#1A365D"),
        alignment=TA_LEFT
    )

    story = []

    # ==========================================
    # PAGE 1: TITLE, META, & ARCHITECTURE
    # ==========================================
    story.append(Paragraph("Automated Citation Verification System", styles["DocTitle"]))
    story.append(Paragraph("Operations Guide: Pipeline Execution, Testing & Real-World Dataset Migration", styles["DocSubtitle"]))
    
    meta_data = [
        [
            Paragraph("<b>Target Audience:</b> Data Engineering, ML, QA & Verification Teams", styles["TableCell"]),
            Paragraph("<b>Corpus Domains:</b> Legal Statutes/Judgments & Scientific Papers", styles["TableCell"])
        ],
        [
            Paragraph("<b>Pipeline Version:</b> 1.0.0-Stage11 (End-to-End Audited)", styles["TableCell"]),
            Paragraph("<b>Test Status:</b> 243/243 Automated Unit & Integration Tests Passing", styles["TableCell"])
        ]
    ]
    meta_table = Table(meta_data, colWidths=[3.5 * inch, 3.5 * inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 0.75, colors.HexColor("#CBD5E0")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 8))

    story.append(create_callout(
        "Executive Summary & Guide Purpose",
        "This operations manual specifies instructions for setting up the environment, running the 243-test automated "
        "test suite, and replacing synthetic test data with the production dataset prepared by the data team. "
        "The system treats the database as an <b>evidence store</b> with complete structural provenance (page coordinates, "
        "exact section hierarchies, table schemas, and parent-child clause links) ensuring high-fidelity claim verification.",
        styles,
        border_color="#319795",
        bg_color="#E6FFFA"
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("1. System Architecture & Lifecycle Stages", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#2B6CB0"), spaceAfter=6))
    
    story.append(Paragraph(
        "The verification engine is architected into three tightly integrated, deterministic stages:",
        styles["Body"]
    ))

    arch_data = [
        [Paragraph("Pipeline Phase", styles["TableHeader"]), Paragraph("Stage Range", styles["TableHeader"]), Paragraph("Primary Function & Key Technologies", styles["TableHeader"])],
        [
            Paragraph("<b>Offline Corpus Building</b>", styles["TableCell"]),
            Paragraph("Stages 0 - 6", styles["TableCellBold"]),
            Paragraph("Document parsing (Docling), structure-aware chunking (legal regex / scientific headings), metadata normalization, domain embeddings (BGE-Large for Legal / SPECTER2 for Scientific), persistent ChromaDB indexing.", styles["TableCell"])
        ],
        [
            Paragraph("<b>Runtime Verification</b>", styles["TableCell"]),
            Paragraph("Stages 7 - 10C", styles["TableCellBold"]),
            Paragraph("Claim/citation detection, bibliography reference resolution, dual-mode retrieval (exact section match + dense vector), source attribution isolation, closed-world LLM semantic verification.", styles["TableCell"])
        ],
        [
            Paragraph("<b>Evaluation & Diagnostics</b>", styles["TableCell"]),
            Paragraph("Stage 11", styles["TableCellBold"]),
            Paragraph("Version-controlled benchmark suites, stage-level error attribution (14 failure categories), automated verdict confusion matrices, and audit report generation.", styles["TableCell"])
        ]
    ]
    arch_table = Table(arch_data, colWidths=[1.8 * inch, 1.1 * inch, 4.1 * inch])
    arch_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2B6CB0")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(arch_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>Core Verification Verdict Taxonomy:</b> Every query produces one of five auditable verdicts: "
        "<b>supported</b> (explicitly proven by evidence), <b>unsupported</b> (evidence mentions context but fails to prove claim), "
        "<b>contradicted</b> (evidence refutes claim), <b>partial</b> (proves subset of multi-part claim), or "
        "<b>not_found</b> (cited statute/reference does not exist).",
        styles["Body"]
    ))

    # Page Break to Page 2
    # story.append(PageBreak())

    # ==========================================
    # PAGE 2: ENVIRONMENT SETUP & TEST SUITE
    # ==========================================
    story.append(Paragraph("2. Environment Setup & Running Existing Code", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#2B6CB0"), spaceAfter=6))

    story.append(Paragraph("2.1 Prerequisites & Virtual Environment Setup", styles["SubSectionHeading"]))
    story.append(Paragraph(
        "The project requires Python 3.11 or 3.12 with PyTorch and Docling. Clone the repository and configure the environment:",
        styles["Body"]
    ))

    setup_code = (
        "# 1. Clone repository\n"
        "git clone https://github.com/dharani20dharan/Citation-Verification.git\n"
        "cd Citation-Verification\n\n"
        "# 2. Create and activate virtual environment\n"
        "# Windows PowerShell:\n"
        "python -m venv .venv\n"
        ".\\.venv\\Scripts\\Activate.ps1\n\n"
        "# Linux / macOS:\n"
        "python3 -m venv .venv\n"
        "source .venv/bin/activate\n\n"
        "# 3. Install dependencies\n"
        "pip install -r requirements.txt"
    )
    story.append(create_code_block(setup_code, styles, "Terminal Commands -- Environment Setup"))
    story.append(Spacer(1, 6))

    story.append(Paragraph("2.2 Environment Configuration (.env)", styles["SubSectionHeading"]))
    story.append(Paragraph(
        "Copy <font face='Courier'>.env.example</font> to <font face='Courier'>.env</font>. "
        "API keys are only necessary when running Stage 10C LLM verification. The corpus builder and evaluation "
        "benchmarks operate fully offline and deterministically.",
        styles["Body"]
    ))

    env_content = (
        "CHROMA_PERSIST_DIR=./vector_db/chroma\n"
        "SCIENTIFIC_EMBEDDING_MODEL=allenai/specter2_base\n"
        "LEGAL_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5\n"
        "EMBEDDING_DEVICE=auto   # Options: auto, cuda, cpu\n"
        "OPENAI_API_KEY=your_key_here          # Optional: Stage 10C LLM semantic verifier\n"
        "GEMINI_API_KEY=your_key_here          # Optional: Stage 10C LLM semantic verifier"
    )
    story.append(create_code_block(env_content, styles, ".env Configuration Template"))
    story.append(Spacer(1, 6))

    story.append(Paragraph("2.3 Running the Automated Test Suite", styles["SubSectionHeading"]))
    story.append(Paragraph(
        "Ensure all baseline tests pass before initiating dataset migration. The project includes 243 automated tests:",
        styles["Body"]
    ))

    test_cmd = (
        "# Run full test suite through virtual environment:\n"
        ".\\.venv\\Scripts\\pytest -v\n\n"
        "# Run domain-specific unit test suites:\n"
        ".\\.venv\\Scripts\\pytest tests/test_legal_chunker.py\n"
        ".\\.venv\\Scripts\\pytest tests/test_scientific_chunker.py\n"
        ".\\.venv\\Scripts\\pytest tests/test_references.py\n"
        ".\\.venv\\Scripts\\pytest tests/test_retrieval.py\n"
        ".\\.venv\\Scripts\\pytest tests/test_integration_pipeline.py"
    )
    story.append(create_code_block(test_cmd, styles, "Running Pytest"))
    story.append(Spacer(1, 6))

    story.append(create_callout(
        "Test Suite Determinism Guarantee",
        "The test suite verifies domain isolation (legal and scientific corpora never mix), exact reference resolution "
        "(e.g., Section 3(d) does not match Section 3(e) or 3(d)(i)), parent-context expansion, and the 14-category error taxonomy. "
        "All 243 tests must show green before proceeding with real dataset integration.",
        styles,
        border_color="#2B6CB0",
        bg_color="#EBF8FF"
    ))

    # Page Break to Page 3
    # story.append(PageBreak())

    # ==========================================
    # PAGE 3: DATASET REPLACEMENT & SCHEMAS
    # ==========================================
    story.append(Paragraph("3. Replacing Synthetic Data with Real-World Dataset", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#2B6CB0"), spaceAfter=6))

    story.append(Paragraph(
        "The repository contains synthetic fixtures in <font face='Courier'>data/</font> for validation. "
        "Follow the mapping below to replace these fixtures with authentic production assets prepared by the data team:",
        styles["Body"]
    ))

    story.append(Paragraph("3.1 Data Directory Mapping", styles["SubSectionHeading"]))
    
    dir_table_data = [
        [Paragraph("Directory Path", styles["TableHeader"]), Paragraph("Current Synthetic Data", styles["TableHeader"]), Paragraph("Real-World Replacement Action", styles["TableHeader"])],
        [
            Paragraph("<font face='Courier'>data/raw/legal/</font>", styles["TableCellBold"]),
            Paragraph("Sample fixture PDFs", styles["TableCell"]),
            Paragraph("Deposit all authentic Legal Judgments & Acts (PDF / DOCX).", styles["TableCell"])
        ],
        [
            Paragraph("<font face='Courier'>data/raw/scientific/</font>", styles["TableCellBold"]),
            Paragraph("Sample publication fixtures", styles["TableCell"]),
            Paragraph("Deposit all authentic scientific research papers (PDF / DOCX).", styles["TableCell"])
        ],
        [
            Paragraph("<font face='Courier'>data/evaluation/</font>", styles["TableCellBold"]),
            Paragraph("17-case synthetic <font face='Courier'>benchmark.jsonl</font>", styles["TableCell"]),
            Paragraph("Replace with real-world curated ground-truth benchmark suite.", styles["TableCell"])
        ],
        [
            Paragraph("<font face='Courier'>vector_db/chroma/</font>", styles["TableCellBold"]),
            Paragraph("Synthetic indexed embeddings", styles["TableCell"]),
            Paragraph("<b>Delete contents completely</b> before re-indexing real dataset.", styles["TableCell"])
        ],
        [
            Paragraph("<font face='Courier'>data/parsed/ & data/chunks/</font>", styles["TableCellBold"]),
            Paragraph("Intermediary stage fixtures", styles["TableCell"]),
            Paragraph("Will be re-populated automatically upon running the pipeline.", styles["TableCell"])
        ]
    ]
    dir_table = Table(dir_table_data, colWidths=[2.1 * inch, 2.0 * inch, 2.9 * inch])
    dir_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2B6CB0")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(dir_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph("3.2 Partner Dataset Integration Modalities", styles["SubSectionHeading"]))
    story.append(Paragraph(
        "Depending on what format the partner team provides, follow the matching procedure:",
        styles["Body"]
    ))
    story.append(Paragraph(
        "&bull; <b>Option A: Raw Documents (PDF / DOCX):</b> Deposit files into <font face='Courier'>data/raw/legal/</font> "
        "and <font face='Courier'>data/raw/scientific/</font>. Run the end-to-end ingestion pipeline described in Section 4.",
        styles["Bullet"]
    ))
    story.append(Paragraph(
        "&bull; <b>Option B: Pre-Chunked JSONL Evidence:</b> If the team has already segmented the documents, files must conform "
        "to <font face='Courier'>EvidenceChunk</font> schema. Save under <font face='Courier'>data/chunks/legal/</font> and "
        "<font face='Courier'>data/chunks/scientific/</font>.",
        styles["Bullet"]
    ))
    story.append(Paragraph(
        "&bull; <b>Option C: Ground-Truth Verification Benchmark:</b> Replace <font face='Courier'>data/evaluation/benchmark.jsonl</font> "
        "with their curated evaluation set containing claims, citation strings, and expected verdicts.",
        styles["Bullet"]
    ))
    story.append(Spacer(1, 6))

    story.append(create_callout(
        "Mandatory Schema Compliance for Pre-Chunked Data",
        "Every record in pre-chunked JSONL must validate against <b>src.models.schemas.EvidenceChunk</b>:<br/>"
        "&bull; <b>chunk_id</b> (str, unique) &nbsp;|&nbsp; <b>document_id</b> (str) &nbsp;|&nbsp; <b>domain</b> ('legal' | 'scientific')<br/>"
        "&bull; <b>title</b> (str) &nbsp;|&nbsp; <b>source</b> (str) &nbsp;|&nbsp; <b>page</b> (int &ge; 1) &nbsp;|&nbsp; <b>section</b> (str) &nbsp;|&nbsp; <b>section_id</b> (str)<br/>"
        "&bull; <b>content_type</b> ('paragraph' | 'section' | 'clause' | 'table' | 'figure' | 'heading' | 'caption')<br/>"
        "&bull; <b>text</b> (clean string &ge; 1 char) &nbsp;|&nbsp; <b>parent_id</b> (str or null if root)",
        styles,
        border_color="#DD6B20",
        bg_color="#FFFAF0"
    ))
    story.append(Spacer(1, 6))

    story.append(Paragraph("3.3 Ground-Truth Benchmark Specification", styles["SubSectionHeading"]))
    story.append(Paragraph(
        "When the team provides the benchmark suite, each line in <font face='Courier'>data/evaluation/benchmark.jsonl</font> "
        "must adhere to this format:",
        styles["Body"]
    ))

    bench_schema = (
        '{\n'
        '  "case_id": "bench_legal_real_001",\n'
        '  "domain": "legal",\n'
        '  "document_id": "patent_act_1970",\n'
        '  "claim": "Section 3(d) requires demonstration of enhanced therapeutic efficacy.",\n'
        '  "reference": "Section 3(d)",\n'
        '  "normalized_reference": "3(d)",\n'
        '  "reference_type": "legal_section",\n'
        '  "expected_reference_status": "exact_match",   # exact_match | not_found | semantic_candidate\n'
        '  "expected_source_status": "available",        # available | unavailable | unresolved_reference\n'
        '  "expected_verdict": "supported",              # supported | unsupported | contradicted | partial | not_found\n'
        '  "expected_conclusive": true,\n'
        '  "expected_evidence_behavior": {\n'
        '    "eligible_required": true,\n'
        '    "required_chunks": ["patent_act_sec_3_d_chunk_01"],\n'
        '    "prohibited_chunks": ["patent_act_sec_3_heading"]\n'
        '  }\n'
        '}'
    )
    story.append(create_code_block(bench_schema, styles, "benchmark.jsonl Record Format"))

    # Page Break to Page 4
    # story.append(PageBreak())

    # ==========================================
    # PAGE 4: PIPELINE RE-RUN & VALIDATION
    # ==========================================
    story.append(Paragraph("4. Step-by-Step Pipeline Re-run with Original Data", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#2B6CB0"), spaceAfter=6))

    story.append(Paragraph(
        "Execute the following sequence in terminal to process raw original documents into indexed Chroma evidence stores:",
        styles["Body"]
    ))

    step_code = (
        "# -------------------------------------------------------------------------\n"
        "# STEP 0: Reset ChromaDB Indexes and Clean Staging Cache\n"
        "# -------------------------------------------------------------------------\n"
        "# Windows PowerShell:\n"
        "Remove-Item -Recurse -Force vector_db\\chroma\\*\n\n"
        "# -------------------------------------------------------------------------\n"
        "# STEP 1: Parse Raw PDF/DOCX using Docling Structural Parser\n"
        "# -------------------------------------------------------------------------\n"
        "python scripts/parse_document.py --input data/raw/scientific/original_paper.pdf --domain scientific\n"
        "python scripts/parse_document.py --input data/raw/legal/original_judgment.pdf --domain legal\n\n"
        "# -------------------------------------------------------------------------\n"
        "# STEP 2: Generate Structure-Aware Evidence Chunks\n"
        "# -------------------------------------------------------------------------\n"
        "python scripts/build_chunks.py --domain scientific\n"
        "python scripts/build_chunks.py --domain legal\n\n"
        "# -------------------------------------------------------------------------\n"
        "# STEP 3: Extract Bibliographic References & Section Headings\n"
        "# -------------------------------------------------------------------------\n"
        "python scripts/build_references.py --domain scientific\n\n"
        "# -------------------------------------------------------------------------\n"
        "# STEP 4: Compute Domain Embeddings (SPECTER2 for Sci / BGE-Large for Legal)\n"
        "# -------------------------------------------------------------------------\n"
        "python scripts/build_embeddings.py --domain scientific\n"
        "python scripts/build_embeddings.py --domain legal\n\n"
        "# -------------------------------------------------------------------------\n"
        "# STEP 5: Index Chunks into Persistent ChromaDB Collections\n"
        "# -------------------------------------------------------------------------\n"
        "python scripts/build_vector_store.py --domain scientific\n"
        "python scripts/build_vector_store.py --domain legal\n\n"
        "# -------------------------------------------------------------------------\n"
        "# ONE-COMMAND AUTOMATED BUILD (Runs Steps 1 through 5 automatically):\n"
        "# -------------------------------------------------------------------------\n"
        "python -m src.pipeline.build_corpus --input data/raw/scientific --domain scientific\n"
        "python -m src.pipeline.build_corpus --input data/raw/legal --domain legal"
    )
    story.append(create_code_block(step_code, styles, "Corpus Building Execution Commands"))
    story.append(Spacer(1, 6))

    story.append(Paragraph("4.1 Auditing the Rebuilt Evidence Corpus", styles["SubSectionHeading"]))
    story.append(Paragraph(
        "Validate database health immediately after indexing to verify chunk persistence and metadata retention:",
        styles["Body"]
    ))

    val_code = (
        "python scripts/validate_corpus.py\n\n"
        "# The validation script performs automated checks:\n"
        "# 1. Connection to persistent ChromaDB (legal_corpus & scientific_corpus)\n"
        "# 2. Validates total chunk record counts and dimensional consistency\n"
        "# 3. Verifies metadata retention (full text, page, section_id, parent_id)\n"
        "# 4. Executes sample retrieval queries to confirm vector health"
    )
    story.append(create_code_block(val_code, styles, "Corpus Validation Execution"))
    story.append(Spacer(1, 6))

    story.append(create_callout(
        "Corpus Validation Gate",
        "Never proceed to Stage 11 benchmark evaluation without a clean passing report from "
        "<b>scripts/validate_corpus.py</b>. If collections show 0 records, verify whether <b>scripts/build_vector_store.py</b> "
        "encountered empty chunk directories or failed write permissions.",
        styles,
        border_color="#E53E3E",
        bg_color="#FFF5F5"
    ))

    # Page Break to Page 5
    # story.append(PageBreak())

    # ==========================================
    # PAGE 5: BENCHMARKING & AUDIT
    # ==========================================
    story.append(Paragraph("5. Testing & Benchmarking on the New Dataset", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#2B6CB0"), spaceAfter=6))

    story.append(Paragraph(
        "Once the original dataset is indexed, run the verification evaluation suite. This tests retrieval precision, "
        "source identity filtering, evidence eligibility, and LLM reasoning accuracy against ground truth.",
        styles["Body"]
    ))

    bench_exec_code = (
        "# 1. Test hybrid retrieval across exact and semantic modes:\n"
        "python scripts/test_retrieval.py --domain legal --query \"Section 3(d)\"\n\n"
        "# 2. Test single claim verification end-to-end:\n"
        "python scripts/verify_claim.py --domain legal --claim \"Supreme Court upheld Section 3(d)\" --ref \"Section 3(d)\"\n\n"
        "# 3. Execute Stage 11 Automated Benchmark & Error Analysis Suite:\n"
        "python scripts/run_benchmark.py --benchmark-file data/evaluation/benchmark.jsonl --output-dir evaluation/"
    )
    story.append(create_code_block(bench_exec_code, styles, "Benchmark Execution Commands"))
    story.append(Spacer(1, 6))

    story.append(Paragraph("5.1 Benchmark Audit Reports", styles["SubSectionHeading"]))
    story.append(Paragraph(
        "Running <font face='Courier'>run_benchmark.py</font> automatically generates two synchronized audit deliverables:",
        styles["Body"]
    ))
    story.append(Paragraph(
        "&bull; <b>evaluation/benchmark_results.json:</b> Machine-readable evaluation report with overall pass rate, "
        "final verdict accuracy, reference resolution accuracy, confusion matrix, stage error counts, and individual case traces.",
        styles["Bullet"]
    ))
    story.append(Paragraph(
        "&bull; <b>evaluation/benchmark_report.md:</b> Executive Markdown audit report complete with confusion matrices, "
        "divergence summaries, and failure breakdown across the 14 error categories.",
        styles["Bullet"]
    ))
    story.append(Spacer(1, 6))

    story.append(Paragraph("5.2 Key Audit Performance Indicators", styles["SubSectionHeading"]))
    
    metrics_data = [
        [Paragraph("Audit Metric", styles["TableHeader"]), Paragraph("Target Threshold", styles["TableHeader"]), Paragraph("Diagnostic Significance", styles["TableHeader"])],
        [
            Paragraph("<b>Overall Pass Rate</b>", styles["TableCellBold"]),
            Paragraph("&ge; 90.0%", styles["TableCellBold"]),
            Paragraph("Percentage of cases satisfying verdict, reference status, source availability, and evidence criteria.", styles["TableCell"])
        ],
        [
            Paragraph("<b>Final Verdict Accuracy</b>", styles["TableCellBold"]),
            Paragraph("&ge; 95.0%", styles["TableCellBold"]),
            Paragraph("Accuracy of final classification across supported, unsupported, contradicted, not_found.", styles["TableCell"])
        ],
        [
            Paragraph("<b>Reference Resolution Accuracy</b>", styles["TableCellBold"]),
            Paragraph("100.0%", styles["TableCellBold"]),
            Paragraph("Exact statutory/citation discovery rate vs rejecting invalid references without false semantic matches.", styles["TableCell"])
        ],
        [
            Paragraph("<b>Source Availability Accuracy</b>", styles["TableCellBold"]),
            Paragraph("100.0%", styles["TableCellBold"]),
            Paragraph("Isolation rate ensuring citing document chunks are never attributed as evidence.", styles["TableCell"])
        ]
    ]
    metrics_table = Table(metrics_data, colWidths=[2.1 * inch, 1.4 * inch, 3.5 * inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2B6CB0")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 6))

    story.append(create_callout(
        "Stage-Level Error Attribution Breakdown",
        "If any case fails in the benchmark, the engine attributes the failure to the earliest diverging stage: "
        "<b>PARSING</b> &rarr; <b>CHUNKING</b> &rarr; <b>CLAIM_EXTRACTION</b> &rarr; <b>REFERENCE_RESOLUTION</b> &rarr; "
        "<b>RETRIEVAL</b> &rarr; <b>SOURCE_ATTRIBUTION</b> &rarr; <b>EVIDENCE_SELECTION</b> &rarr; <b>SEMANTIC_VERIFICATION</b>. "
        "This isolates whether errors stem from data ingestion, retrieval ranking, or LLM reasoning.",
        styles,
        border_color="#805AD5",
        bg_color="#FAF5FF"
    ))

    # Page Break to Page 6
    # story.append(PageBreak())

    # ==========================================
    # PAGE 6: TROUBLESHOOTING & CHECKLIST
    # ==========================================
    story.append(Paragraph("6. Troubleshooting & Migration Checklist", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#2B6CB0"), spaceAfter=6))

    story.append(Paragraph("6.1 Common Pitfalls & Diagnostic Resolutions", styles["SubSectionHeading"]))

    pitfalls_data = [
        [Paragraph("Issue / Symptom", styles["TableHeader"]), Paragraph("Root Cause", styles["TableHeader"]), Paragraph("Resolution Procedure", styles["TableHeader"])],
        [
            Paragraph("<b>Chroma Lock / Collision</b>", styles["TableCellBold"]),
            Paragraph("Previous process did not release ChromaDB directory lock.", styles["TableCell"]),
            Paragraph("Terminate all Python background processes. Delete <font face='Courier'>vector_db/chroma/</font> before fresh indexing.", styles["TableCell"])
        ],
        [
            Paragraph("<b>CUDA Out of Memory</b>", styles["TableCellBold"]),
            Paragraph("Batch size too large during embedding computation.", styles["TableCell"]),
            Paragraph("Set <font face='Courier'>EMBEDDING_DEVICE=cpu</font> in <font face='Courier'>.env</font> or reduce batch size in <font face='Courier'>scripts/build_embeddings.py</font>.", styles["TableCell"])
        ],
        [
            Paragraph("<b>Heading Evidence Leak</b>", styles["TableCellBold"]),
            Paragraph("Structural section heading selected as substantive claim evidence.", styles["TableCell"]),
            Paragraph("Verify chunk's <font face='Courier'>content_type</font> is marked <font face='Courier'>heading</font>. The verifier automatically rejects headings as independent proof.", styles["TableCell"])
        ],
        [
            Paragraph("<b>Missing Reference Hallucination</b>", styles["TableCellBold"]),
            Paragraph("Semantic search returns closest vector for nonexistent section.", styles["TableCell"]),
            Paragraph("The exact statutory resolver prevents semantic fallback when reference lookup fails. Ensure <font face='Courier'>expected_reference_status=not_found</font>.", styles["TableCell"])
        ]
    ]
    pitfalls_table = Table(pitfalls_data, colWidths=[1.8 * inch, 2.2 * inch, 3.0 * inch])
    pitfalls_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2B6CB0")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(pitfalls_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph("6.2 Team Handoff & Migration Verification Checklist", styles["SubSectionHeading"]))
    story.append(Paragraph(
        "Complete this verification checklist when receiving the original dataset from the partner team:",
        styles["Body"]
    ))

    checklist_items = [
        "[  ] <b>Step 1: Backup Baseline:</b> Archive current synthetic fixtures (<font face='Courier'>data/</font>) and benchmark results.",
        "[  ] <b>Step 2: File Placement:</b> Deposit real PDF/DOCX files into <font face='Courier'>data/raw/legal/</font> and <font face='Courier'>data/raw/scientific/</font>.",
        "[  ] <b>Step 3: Benchmark Placement:</b> Replace <font face='Courier'>data/evaluation/benchmark.jsonl</font> with the real benchmark suite.",
        "[  ] <b>Step 4: Vector DB Reset:</b> Clear <font face='Courier'>vector_db/chroma/</font> to eliminate stale embeddings.",
        "[  ] <b>Step 5: Run Ingestion:</b> Execute <font face='Courier'>build_corpus</font> for both legal and scientific domains.",
        "[  ] <b>Step 6: Validate Corpus:</b> Run <font face='Courier'>python scripts/validate_corpus.py</font> and verify non-zero record counts.",
        "[  ] <b>Step 7: Run Unit Tests:</b> Run <font face='Courier'>pytest</font> to ensure parser and chunker logic remain compatible.",
        "[  ] <b>Step 8: Run Stage 11 Benchmark:</b> Execute <font face='Courier'>python scripts/run_benchmark.py</font> and inspect pass rate.",
        "[  ] <b>Step 9: Review Audit Reports:</b> Review <font face='Courier'>evaluation/benchmark_report.md</font> and diagnose any stage errors."
    ]

    for item in checklist_items:
        story.append(Paragraph(item, styles["Bullet"]))

    story.append(Spacer(1, 10))
    story.append(create_callout(
        "Repository & Engineering Support",
        "For schema adjustments or parser rule extensions on complex legal formats (e.g. state amendment acts) "
        "or scientific publisher formats, refer to <b>src.parsing.docling_parser</b> and <b>src.chunking.legal_chunker</b>. "
        "All code is under version control at: <b>github.com/dharani20dharan/Citation-Verification</b>.",
        styles,
        border_color="#2B6CB0",
        bg_color="#EBF8FF"
    ))

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated clean 6-page PDF guide at: {output_path.resolve()}")


if __name__ == "__main__":
    out_file = Path("Citation_Verification_Execution_and_Dataset_Migration_Guide.pdf")
    build_pdf(out_file)
