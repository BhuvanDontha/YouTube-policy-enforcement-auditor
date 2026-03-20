"""
Policy Enforcement Auditor — Streamlit Application
4 views: Live Classifier, Consistency Audit, Disagreement Dashboard, System Evaluation

Run: streamlit run app/streamlit_app.py

Author: Bhuvan Dontha
"""

import os
import sys
import json
import streamlit as st
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from classifiers.rules_classifier import classify_rules

# Try to import LLM classifier (may not have API key)
LLM_AVAILABLE = False
try:
    from classifiers.llm_classifier import LLMClassifier
    if os.getenv("GEMINI_API_KEY"):
        LLM_AVAILABLE = True
except Exception:
    pass

# Try to import YouTube video processor
YT_AVAILABLE = False
try:
    from data.youtube_transcript import get_transcript, summarize_for_classification, summarize_youtube_direct
    YT_AVAILABLE = True
except Exception:
    pass

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Policy Enforcement Auditor",
    page_icon="\U0001F6E1",
    layout="wide",
)

st.title("\U0001F6E1 Policy Enforcement Auditor")
st.caption("AI-Powered Content Policy Classification & Enforcement Consistency Analysis")

# ==================== PASSWORD GATE ====================
app_password = os.environ.get("APP_PASSWORD", "")
if app_password:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        password = st.text_input("Enter access code to continue", type="password")
        if password == app_password:
            st.session_state.authenticated = True
            st.rerun()
        elif password:
            st.error("Incorrect access code.")
        st.stop()

# ==================== SIDEBAR ====================
view = st.sidebar.radio(
    "Select View",
    ["\U0001F50D Live Classifier", "\U0001F4CA Disagreement Dashboard",
     "\u2696 Consistency Audit", "\U0001F4C8 System Evaluation"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Data Source:** YouTube Advertiser-Friendly Content Guidelines")
st.sidebar.markdown("[View Guidelines](https://support.google.com/youtube/answer/6162278)")
st.sidebar.markdown("---")
st.sidebar.markdown("Built by **Bhuvan Dontha**")


# ==================== VIEW 1: LIVE CLASSIFIER ====================
if view == "\U0001F50D Live Classifier":
    st.header("Live Content Classifier")
    st.markdown(
        "Enter a content description or paste a YouTube URL to auto-extract the transcript. "
        "The system classifies it using both an LLM and a rules-based baseline, then highlights divergence."
    )

    # Input mode selection
    input_mode = st.radio("Input method", ["Text description", "YouTube URL"], horizontal=True)

    user_input = ""

    if input_mode == "YouTube URL":
        if not YT_AVAILABLE:
            st.warning("YouTube module not available. Check data/youtube_transcript.py")
        else:
            yt_url = st.text_input(
                "YouTube Video URL",
                placeholder="https://www.youtube.com/watch?v=...",
            )
            if yt_url:
                # Primary: One-shot Gemini (processes video directly, no scraping)
                if LLM_AVAILABLE:
                    with st.spinner("Analyzing video with Gemini (native video understanding)..."):
                        try:
                            user_input = summarize_youtube_direct(yt_url)
                            st.success("Video analyzed via Gemini native video processing")
                            st.info(f"**Content summary:** {user_input}")
                        except Exception as e:
                            st.warning(f"Gemini direct analysis failed ({e}). Trying transcript extraction...")
                            # Fallback: transcript extraction + summarization
                            result = get_transcript(yt_url, max_chars=100000)
                            if "error" in result:
                                st.error(f"Could not fetch transcript: {result['error']}")
                            else:
                                st.success(
                                    f"Transcript fetched via {result['source']} — "
                                    f"{result['char_count']} chars"
                                    f"{' (truncated)' if result['truncated'] else ''}"
                                )
                                with st.expander("View raw transcript"):
                                    st.text(result["full_text"][:2000])
                                try:
                                    user_input = summarize_for_classification(result["full_text"])
                                    st.info(f"**Content summary:** {user_input}")
                                except Exception as e2:
                                    user_input = result["full_text"][:1000]
                else:
                    # No Gemini key — try transcript API only
                    with st.spinner("Fetching transcript..."):
                        result = get_transcript(yt_url, max_chars=100000)
                    if "error" in result:
                        st.error(f"Could not fetch transcript: {result['error']}")
                    else:
                        user_input = result["full_text"][:1000]
                        st.success(f"Transcript fetched — {result['char_count']} chars")
    else:
        user_input = st.text_area(
            "Content Description",
            placeholder="e.g., Gaming video with graphic beheading scene in the first 10 seconds",
            height=100,
        )

    if st.button("Classify", type="primary") and user_input:
        col1, col2 = st.columns(2)

        # Rules classifier
        with col1:
            st.subheader("\U0001F4CF Rules Classifier")
            rules_result = classify_rules(user_input)
            for r in rules_result:
                color = {"RED": "red", "YELLOW": "orange", "GREEN": "green"}.get(r["severity_tier"], "gray")
                st.markdown(f"**{r['policy_name']}** — :{color}[{r['severity_tier']}]")
                st.caption(r["reasoning"])

        # LLM classifier
        with col2:
            st.subheader("\U0001F916 LLM Classifier (Gemini)")
            if LLM_AVAILABLE:
                with st.spinner("Classifying with Gemini..."):
                    llm = LLMClassifier()
                    llm_result = llm.classify(user_input)
                for r in llm_result:
                    color = {"RED": "red", "YELLOW": "orange", "GREEN": "green"}.get(r["severity_tier"], "gray")
                    st.markdown(f"**{r['policy_name']}** — :{color}[{r['severity_tier']}] ({r['confidence']})")
                    st.caption(r["reasoning"])
            else:
                st.warning("GEMINI_API_KEY not set. Set it to enable LLM classification.")
                st.caption("Get a free key at: https://aistudio.google.com/apikey")

        # Disagreement check
        st.markdown("---")
        rules_primary = rules_result[0] if rules_result else {}
        llm_primary = llm_result[0] if LLM_AVAILABLE and llm_result else {}

        if rules_primary and llm_primary:
            if rules_primary.get("policy_name") != llm_primary.get("policy_name"):
                st.error(
                    f"\u26A0 **DISAGREEMENT DETECTED** — Rules says **{rules_primary.get('policy_name', 'None')}** "
                    f"but LLM says **{llm_primary.get('policy_name', 'None')}**. "
                    f"This content should be flagged for human review."
                )
            elif rules_primary.get("severity_tier") != llm_primary.get("severity_tier"):
                st.warning(
                    f"\u26A0 **SEVERITY DISAGREEMENT** — Both say **{rules_primary.get('policy_name')}** "
                    f"but Rules says **{rules_primary.get('severity_tier')}** while LLM says **{llm_primary.get('severity_tier')}**."
                )
            else:
                st.success("\u2705 **AGREEMENT** — Both classifiers agree on policy and severity.")


# ==================== VIEW 2: DISAGREEMENT DASHBOARD ====================
elif view == "\U0001F4CA Disagreement Dashboard":
    st.header("Disagreement Dashboard")
    st.markdown(
        "Shows where the LLM and rules classifiers disagree. "
        "Disagreement zones are where human reviewers add the most value."
    )

    summary_path = os.path.join(OUTPUT_DIR, "ensemble_summary.json")
    disagree_path = os.path.join(OUTPUT_DIR, "disagreements.csv")

    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)

        # Top-level metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Descriptions", summary.get("total_descriptions", 0))
        col2.metric("Disagreements", summary.get("total_disagreements", 0))
        col3.metric("Disagreement Rate", f"{summary.get('disagreement_rate', 0)}%")

        # Type breakdown
        st.subheader("Disagreement Type Breakdown")
        type_data = summary.get("type_breakdown", {})
        if type_data:
            df_types = pd.DataFrame([{"Type": k, "Count": v} for k, v in type_data.items()])
            st.bar_chart(df_types.set_index("Type"))

        # Policy-level disagreements
        st.subheader("Disagreement by Policy Category")
        policy_data = summary.get("policy_disagreement_counts", {})
        if policy_data:
            df_policy = pd.DataFrame([{"Policy": k, "Disagreements": v} for k, v in policy_data.items()])
            st.bar_chart(df_policy.set_index("Policy"))

        # Top priority cases
        st.subheader("Top 10 Human Review Priority Cases")
        top_cases = summary.get("top_priority_cases", [])
        if top_cases:
            st.dataframe(pd.DataFrame(top_cases), use_container_width=True)

    else:
        st.info("Run the pipeline first: `python run_pipeline.py`")

    # Full disagreements table
    if os.path.exists(disagree_path):
        with st.expander("View All Disagreements"):
            df = pd.read_csv(disagree_path)
            st.dataframe(df[df["is_disagreement"] == True], use_container_width=True)


# ==================== VIEW 3: CONSISTENCY AUDIT ====================
elif view == "\u2696 Consistency Audit":
    st.header("Enforcement Consistency Audit")
    st.markdown(
        "Tests whether semantically similar content descriptions receive "
        "the same classification. Inconsistencies indicate fairness risks."
    )

    audit_path = os.path.join(OUTPUT_DIR, "consistency_audit.json")
    if os.path.exists(audit_path):
        with open(audit_path, "r") as f:
            audit = json.load(f)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Pairs Tested", audit.get("total_pairs", 0))
        col2.metric("Policy Consistency", f"{audit.get('policy_consistency_rate', 0)}%")
        col3.metric("Full Consistency", f"{audit.get('full_consistency_rate', 0)}%")

        # Per-policy consistency
        st.subheader("Consistency Rate by Policy Category")
        per_policy = audit.get("per_policy", {})
        if per_policy:
            df_pp = pd.DataFrame([
                {"Policy": k, "Consistency Rate (%)": v["consistency_rate"], "Pairs": v["total_pairs"]}
                for k, v in per_policy.items()
            ]).sort_values("Consistency Rate (%)")
            st.bar_chart(df_pp.set_index("Policy")["Consistency Rate (%)"])

        # Inconsistent examples
        st.subheader("Sample Inconsistencies")
        examples = audit.get("inconsistent_examples", [])
        if examples:
            for ex in examples[:5]:
                with st.expander(f"{ex.get('content_id_a', '')} vs {ex.get('content_id_b', '')}"):
                    st.markdown(f"**A:** {ex.get('description_a', '')}")
                    st.markdown(f"  Predicted: {ex.get('predicted_policy_a', '')} ({ex.get('predicted_severity_a', '')})")
                    st.markdown(f"**B:** {ex.get('description_b', '')}")
                    st.markdown(f"  Predicted: {ex.get('predicted_policy_b', '')} ({ex.get('predicted_severity_b', '')})")
                    st.markdown(f"  True policy: **{ex.get('true_policy', '')}**")
        else:
            st.success("All tested pairs are fully consistent.")
    else:
        st.info("Run the pipeline first: `python run_pipeline.py`")


# ==================== VIEW 4: SYSTEM EVALUATION ====================
elif view == "\U0001F4C8 System Evaluation":
    st.header("System Evaluation Metrics")
    st.markdown(
        "Precision, recall, and F1 per policy category. "
        "Evaluated against ground truth labels."
    )

    eval_path = os.path.join(OUTPUT_DIR, "evaluation_report.json")
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            eval_data = json.load(f)

        # Top-level metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Policy Accuracy", f"{eval_data.get('policy_accuracy', 0)}%")
        col2.metric("Severity Accuracy", f"{eval_data.get('severity_accuracy', 0)}%")
        col3.metric("Macro F1", f"{eval_data.get('macro_f1', 0)}")
        col4.metric("Total Evaluated", eval_data.get("total_evaluated", 0))

        # Per-category table
        st.subheader("Per-Category Performance")
        per_cat = eval_data.get("per_category", {})
        if per_cat:
            df_cat = pd.DataFrame([
                {"Category": k, **v} for k, v in per_cat.items()
            ]).sort_values("f1_score", ascending=False)
            st.dataframe(df_cat, use_container_width=True)

            # F1 bar chart
            st.subheader("F1 Score by Category")
            chart_data = df_cat.set_index("Category")["f1_score"]
            st.bar_chart(chart_data)

        # Confusion matrix
        st.subheader("Confusion Matrix")
        confusion = eval_data.get("confusion_matrix", {})
        if confusion:
            all_labels = sorted(set(list(confusion.keys()) + [
                v for inner in confusion.values() for v in inner.keys()
            ]))
            matrix = []
            for true_label in all_labels:
                row = {"True \\ Predicted": true_label}
                for pred_label in all_labels:
                    row[pred_label] = confusion.get(true_label, {}).get(pred_label, 0)
                matrix.append(row)
            st.dataframe(pd.DataFrame(matrix).set_index("True \\ Predicted"), use_container_width=True)
    else:
        st.info("Run the pipeline first: `python run_pipeline.py`")
