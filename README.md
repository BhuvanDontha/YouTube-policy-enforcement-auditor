# 🛡 Policy Enforcement Auditor

**An Independent Evaluation Framework for Content Policy Classification**

> Content platforms enforce policies at scale using automated classifiers. This project builds **evaluation infrastructure** for those systems — the same way software engineering has test suites, this is a test suite for content classification.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io)
[![Gemini](https://img.shields.io/badge/Gemini_2.5-Flash-4285F4.svg)](https://ai.google.dev)
[![License](https://img.shields.io/badge/License-Educational-green.svg)](#license)

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [The Dual-Classifier Approach](#the-dual-classifier-approach)
- [YouTube Video Analysis](#youtube-video-analysis)
- [Quick Start (Step-by-Step)](#quick-start-step-by-step)
- [Using the Dashboard](#using-the-dashboard)
- [Key Metrics Explained](#key-metrics-explained)
- [Data Source](#data-source)
- [Limitations & Design Choices](#limitations--design-choices)
- [Skills Demonstrated](#skills-demonstrated)
- [Author](#author)

---

## Problem Statement

Automated content classification operates across dozens of policy categories, each with multiple severity tiers. A single global accuracy metric hides per-category variation — a system at 90% overall might be 98% on clear-cut categories and 65% on contextually complex ones.

**This project builds an independent evaluation framework** that:

1. Classifies content using two fundamentally different approaches (LLM + deterministic rules baseline)
2. Analyzes patterns of agreement and divergence between them
3. Measures classification consistency across semantically similar content
4. Produces per-policy-category evaluation metrics (precision, recall, F1)

The output is a **classification quality dashboard** — showing which categories are straightforward to classify, which require more nuanced handling, and where different classification methodologies produce different results.

**Why two classifiers?** A single classifier cannot evaluate itself. Two independent approaches with different strengths create a cross-validation signal. Where they agree, classification is high-confidence. Where they diverge, the content is contextually complex. That divergence pattern maps the **difficulty landscape** of a policy taxonomy.

---

## How It Works

The system follows a simple pipeline:

```
                        ┌──────────────────────┐
                        │   Content Input       │
                        │ (text or YouTube URL) │
                        └──────────┬───────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                              │
           ┌───────▼───────┐            ┌──────────▼──────────┐
           │ Rules Classifier│           │  LLM Classifier     │
           │ (Keywords)      │           │  (Gemini 2.5 Flash) │
           └───────┬────────┘           └──────────┬──────────┘
                    │                              │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Cross-Classifier Analysis  │
                    │   (Agreement vs Divergence)  │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                     │
    ┌─────────▼────────┐  ┌───────▼────────┐  ┌────────▼───────┐
    │ Divergence        │  │ Consistency    │  │ Evaluation     │
    │ Dashboard         │  │ Audit          │  │ Metrics        │
    └──────────────────┘  └────────────────┘  └────────────────┘
```

**For text input:** You type a content description (e.g., "Gaming video with graphic violence") and both classifiers analyze it side by side.

**For YouTube URLs:** The system uses Gemini's native video understanding to analyze the actual video — audio, visuals, speech, and text overlays — performing a structured audit against all 14 policy categories. No scraping. No transcript extraction needed. Works with any language.

---

## Architecture

```
policy-enforcement-auditor/
├── data/
│   ├── policy_taxonomy.json          # 14 categories × 3 tiers (public YouTube guidelines)
│   ├── generate_data.py              # Synthetic content description generator
│   ├── youtube_transcript.py         # YouTube video analysis (Gemini native + transcript fallback)
│   ├── synthetic_content.csv         # 500 labeled descriptions
│   └── ground_truth.csv              # Ground truth labels for evaluation
│
├── classifiers/
│   ├── llm_classifier.py             # Gemini 2.5 Flash structured prompting
│   ├── rules_classifier.py           # Keyword + regex deterministic baseline
│   └── ensemble.py                   # Cross-classifier analysis engine
│
├── analysis/
│   ├── metrics.py                    # Precision, recall, F1 per category
│   └── consistency_audit.py          # Same input → same output?
│
├── app/
│   └── streamlit_app.py              # Interactive dashboard (4 views + YouTube analysis)
│
├── outputs/                           # Generated by run_pipeline.py
│   ├── llm_classifications.csv       # LLM classification results for all 500 descriptions
│   ├── disagreements.csv             # Full LLM vs rules comparison
│   ├── ensemble_summary.json         # Divergence statistics
│   ├── evaluation_report.json        # Accuracy metrics (JSON)
│   ├── evaluation_report.txt         # Accuracy metrics (readable)
│   └── consistency_audit.json        # Consistency analysis
│
├── run_pipeline.py                    # Full pipeline orchestrator (runs all 6 steps)
├── requirements.txt                   # Python dependencies
├── .env.example                       # API key template
└── README.md
```

---

## The Dual-Classifier Approach

| Classifier | How It Works | Strength | Limitation |
|---|---|---|---|
| **LLM (Gemini 2.5 Flash)** | Structured prompting with full policy taxonomy as context. Returns policy, severity, confidence, and reasoning chain for each classification. | Understands context, nuance, sarcasm, multi-policy content. Can classify in any language. | May over-reason or produce inconsistent outputs across runs |
| **Rules (Keywords)** | Deterministic keyword matching against policy tier descriptions. Scans for explicit words like "gun," "nude," "scam," etc. | Fast, reproducible, transparent logic. Same input always produces same output. | Cannot reason about context or implied meaning. Misses nuanced content. |

### Cross-Classifier Divergence Patterns

| Pattern | What It Indicates | Analytical Value |
|---|---|---|
| Both agree on policy + severity | High-confidence classification zone | Baseline — content is unambiguous |
| LLM detects, rules miss | Content requires contextual understanding beyond keywords | Maps where LLM adds value over simple automation |
| Rules detect, LLM misses | Explicit signals present but LLM reasoned past them | Identifies areas for prompt improvement |
| Different policies entirely | Content is genuinely multi-faceted or ambiguous | Highest contextual complexity — most informative for evaluation |

### The 14 Policy Categories

Based on YouTube's publicly documented [Advertiser-Friendly Content Guidelines](https://support.google.com/youtube/answer/6162278):

| # | Category | What It Covers |
|---|---|---|
| 1 | Inappropriate Language | Profanity, slurs, strong language |
| 2 | Violence | Fighting, weapons, blood, injury |
| 3 | Adult Content | Sexual content, nudity, suggestive material |
| 4 | Shocking Content | Graphic, disturbing, or gory content |
| 5 | Harmful Acts & Unreliable Content | Dangerous stunts, misinformation, unverified claims |
| 6 | Hateful & Derogatory Content | Hate speech, discrimination, slurs |
| 7 | Recreational Drugs | Drug use, drug references, substance glorification |
| 8 | Firearms | Guns, weapons modification, weapons sales |
| 9 | Controversial Issues | Political, social, religious controversy |
| 10 | Sensitive Events | War, terrorism, disasters, mass casualties |
| 11 | Enabling Dishonest Behavior | Scams, fraud, hacking, deception |
| 12 | Inappropriate Content for Kids | Content targeting or involving minors inappropriately |
| 13 | Incendiary & Demeaning | Public shaming, bullying, humiliation |
| 14 | Tobacco-Related Content | Tobacco use, smoking, vaping, promotion |

Each category has three severity tiers: **GREEN** (minimal concern), **YELLOW** (moderate concern), **RED** (high concern).

---

## YouTube Video Analysis

The dashboard supports direct YouTube URL input. Instead of relying on transcript scraping (which gets blocked on cloud deployments), the system uses **Gemini's native video understanding** to analyze the actual video.

### How It Works

1. You paste a YouTube URL
2. The system fetches the video thumbnail and title via YouTube's oEmbed API
3. Gemini 2.5 Flash receives the video URL and performs a **structured 14-category audit** — analyzing audio, speech, visuals, text overlays, and context
4. For each of the 14 policy categories, Gemini reports: `PRESENT` (with specific evidence) or `ABSENT`
5. The findings are assembled into a content description and sent to both classifiers

### Why Structured Audit Instead of Free-Form Summary?

Free-form descriptions get sanitized by LLM safety filters. Asking "describe this video" produces soft outputs like "a man exploring streets." The structured audit format (YES/NO per category with evidence) treats the task as **analytical classification** rather than content generation, which safety filters handle differently.

### Multi-Language Support

Gemini natively processes video in any language — Hindi, Tamil, Telugu, Spanish, Arabic, Japanese, etc. The audit results are always returned in English regardless of the video's original language. No translation libraries needed.

---

## Quick Start (Step-by-Step)

### Prerequisites

- **Python 3.10+** installed on your machine
- A **Gemini API key** (free) from [Google AI Studio](https://aistudio.google.com/apikey)
- Basic familiarity with terminal / command line

### Step 1: Clone the Repository

```bash
git clone https://github.com/BhuvanDontha/policy-enforcement-auditor.git
cd policy-enforcement-auditor
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs: `streamlit`, `pandas`, `google-genai`, `youtube-transcript-api`, and other required packages.

### Step 4: Set Up Your Gemini API Key

Get a free API key from [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey). Then:

```bash
# macOS / Linux
export GEMINI_API_KEY='your-api-key-here'

# Windows (Command Prompt)
set GEMINI_API_KEY=your-api-key-here

# Windows (PowerShell)
$env:GEMINI_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```bash
cp .env.example .env
# Edit .env and paste your key
```

### Step 5: Run the Full Pipeline (One-Time Setup)

This classifies all 500 synthetic descriptions, runs the cross-classifier analysis, computes evaluation metrics, and generates all output files:

```bash
python run_pipeline.py
```

**What happens:**
1. Loads 500 synthetic content descriptions
2. Runs rules-based classifier on all 500
3. Runs LLM classifier (Gemini) on all 500 (~3-5 minutes)
4. Runs cross-classifier analysis (disagreement detection)
5. Evaluates accuracy against ground truth
6. Runs consistency audit
7. Saves all results to `outputs/`

### Step 6: Launch the Dashboard

```bash
streamlit run app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser. You should see the dashboard with 4 views.

### Step 7: Try Classifying a YouTube Video

1. In the dashboard, select **"Live Classifier"** from the sidebar
2. Switch input mode to **"YouTube URL"**
3. Paste any YouTube video URL
4. The system will show the video thumbnail, title, and channel
5. Gemini analyzes the video and produces a structured content audit
6. Both classifiers run on the audit results
7. Divergence between classifiers is highlighted

---

## Using the Dashboard

### View 1: Live Classifier

The main interactive view. Two input modes:

**Text Description Mode:**
- Type or paste any content description
- Click "Classify" to see side-by-side results from both classifiers
- Divergence between them is highlighted in red (different policies) or yellow (different severity)

**YouTube URL Mode:**
- Paste a YouTube video URL
- The system shows thumbnail, title, and channel name
- Gemini performs a structured 14-category audit of the video
- Results show which policy categories were detected with specific evidence
- Both classifiers then analyze the audit findings

### View 2: Disagreement Dashboard

Shows aggregate analysis across all 500 synthetic descriptions:
- **Total divergence rate** — what percentage of content produces different classifications
- **Divergence by type** — breakdown of the 4 divergence patterns
- **Divergence by policy category** — which categories have the most disagreement
- **Top 10 priority cases** — the most contextually complex content descriptions

### View 3: Consistency Audit

Tests whether similar content descriptions get the same classification:
- **Per-category consistency scores** — how reproducible is classification within each category
- **Inconsistency examples** — specific content pairs that received different classifications despite belonging to the same policy category
- Low consistency indicates areas where classification is sensitive to input phrasing

### View 4: System Evaluation

Accuracy metrics for the LLM classifier against ground truth labels:
- **Policy accuracy** — percentage of correct policy category assignments
- **Severity accuracy** — percentage where both policy AND severity tier match
- **Macro F1** — balanced precision/recall across all 14 categories
- **Per-category performance table** — precision, recall, F1, TP, FP, FN for each category
- **Confusion matrix** — shows which categories get confused with each other

---

## Key Metrics Explained

| Metric | What It Measures | How to Read It |
|---|---|---|
| **Divergence Rate** | % of content where two classifiers disagree | Higher = more contextually complex content in that category |
| **Complexity Score** | Severity-weighted divergence score | Higher = content requires more nuanced classification |
| **Policy Accuracy** | % correct policy category (LLM vs ground truth) | Our result: 75.8% on zero-shot classification |
| **Severity Accuracy** | % correct policy AND severity tier | Our result: 68.4% — severity is harder than category |
| **Consistency Rate** | % of similar content pairs with identical classification | 84% overall; Violence at 30% indicates high sensitivity |
| **Macro F1** | Balanced precision/recall across all categories | Our result: 0.564 — strong for zero-shot, room to improve |

### Per-Category Highlights from Our Results

| Category | F1 Score | Interpretation |
|---|---|---|
| Inappropriate Language | 0.968 | Keyword-driven, nearly perfect |
| Recreational Drugs | 0.935 | Domain-specific vocabulary, unambiguous |
| Enabling Dishonest Behavior | 0.909 | LLM understands intent well |
| Adult Content | 0.857 | High precision (0.975), moderate recall |
| Violence | 0.774 | High recall (0.828) but lower precision — over-classifies |
| Harmful Acts | 0.635 | Broad catch-all category, high false positive rate |
| Incendiary & Demeaning | 0.500 | Perfect precision but low recall — LLM is too conservative |

---

## Data Source

All synthetic data is generated from YouTube's publicly documented **Advertiser-Friendly Content Guidelines**:

- **Source:** [support.google.com/youtube/answer/6162278](https://support.google.com/youtube/answer/6162278)
- **14 policy categories** × 3 severity tiers (GREEN / YELLOW / RED)
- **500 content descriptions**, each traceable to a specific guideline example
- **Zero proprietary data.**

The YouTube video analysis feature uses Gemini's native video understanding to analyze real public videos in real-time.

---

## Limitations & Design Choices

**Synthetic data, not production signals.** The 500 synthetic descriptions are clean text derived from public guidelines. Production classifiers ingest video metadata, thumbnail analysis, audio transcription, frame-level visuals, and engagement signals. Evaluation metrics here represent upper-bound performance on clean inputs.

**YouTube analysis depends on Gemini's video processing.** The structured audit is only as good as Gemini's ability to perceive the video content. Some nuances in audio tone, visual context, or cultural references may be missed.

**Rules classifier is intentionally weak.** It uses simple keyword matching and is designed to be a baseline, not a production-grade classifier. The value is in the *comparison* between classifiers, not in either classifier individually.

**Why the framework is still valuable:** The core contribution is the **evaluation infrastructure**, not the classifiers. The cross-classifier analysis, consistency audit, and metrics pipeline work identically regardless of input source. In a production context, swap the classifiers with any models you want to evaluate. The framework is modular and portable.

```
CURRENT (this project):
  Content → LLM + Rules → Evaluation Framework

PRODUCTION (any platform):
  Content → Any Classifier A + Any Classifier B → Same Evaluation Framework
```

---

## Author

**Bhuvan Dontha**
Trust & Safety Analytics | YouTube Monetization
[LinkedIn](https://www.linkedin.com/in/bhuvan-dontha-838590218/) · [GitHub](https://github.com/BhuvanDontha)

---

## License

This project is for educational and portfolio purposes. All synthetic data is generated from publicly available guidelines. The YouTube Advertiser-Friendly Content Guidelines referenced in this project are publicly available at [support.google.com/youtube/answer/6162278](https://support.google.com/youtube/answer/6162278).
