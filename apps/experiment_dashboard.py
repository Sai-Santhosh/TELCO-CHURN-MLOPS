"""
Experiment Dashboard - BCG-style Retention Decision Studio.

Streamlit app with pages: Design, Run Demo, Results, Recommendation.
"""

import json
import sys
from pathlib import Path

# Add project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(page_title="Retention Decision Studio", page_icon="📊", layout="wide")

# Custom CSS for clean visuals
st.markdown("""
<style>
    .metric-card { background: #f8f9fa; padding: 16px; border-radius: 8px; margin: 8px 0; border-left: 4px solid #1a5276; }
    .recommendation-ship { background: #d5f5e3; padding: 12px; border-radius: 6px; }
    .recommendation-hold { background: #fdebd0; padding: 12px; border-radius: 6px; }
    .recommendation-iterate { background: #ebf5fb; padding: 12px; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

def get_experiment_ids():
    """List experiment IDs from data/experiments."""
    exp_dir = ROOT / "data" / "experiments"
    if not exp_dir.exists():
        return []
    return [d.name for d in exp_dir.iterdir() if d.is_dir()]

def load_latest_analysis():
    """Load most recent analysis from artifacts."""
    art_dir = ROOT / "artifacts" / "experiments"
    if not art_dir.exists():
        return None, None
    exps = [d for d in art_dir.iterdir() if d.is_dir()]
    if not exps:
        return None, None
    latest = max(exps, key=lambda p: (p / "analysis.json").stat().st_mtime if (p / "analysis.json").exists() else 0)
    if not (latest / "analysis.json").exists():
        return None, None
    with open(latest / "analysis.json") as f:
        return latest.name, json.load(f)

def main():
    st.title("📊 Experimentation & Retention Decision Studio")
    st.caption("BCG-style client-facing decision product")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Design", "Run Demo", "Results", "Recommendation"])
    
    with tab1:
        st.header("Experiment Design")
        exp_id = st.text_input("Experiment ID", value="retention_2025_q1")
        allocation = st.slider("Treatment allocation", 0.2, 0.8, 0.5, 0.05)
        primary_metric = st.selectbox("Primary metric", ["churn_rate", "retention_rate"])
        
        st.subheader("Power / MDE Calculator")
        baseline = st.number_input("Baseline rate (e.g., churn)", 0.01, 0.99, 0.265, 0.01)
        mde_rel = st.slider("Target MDE (relative)", 0.05, 0.5, 0.15, 0.05)
        try:
            from src.experimentation.stats import sample_size_proportion, mde_proportion
            n_needed = sample_size_proportion(baseline, mde_rel)
            st.info(f"Sample size needed per arm: **{n_needed:,}** (for 80% power, α=0.05)")
            mde_ach = mde_proportion(baseline, 500, allocation=allocation)
            st.info(f"With 500/arm, detectable MDE: **{mde_ach*100:.1f}%** relative")
        except Exception as e:
            st.warning(f"Power calc: {e}")
    
    with tab2:
        st.header("Run Demo")
        demo_exp_id = st.text_input("Experiment ID for demo", value="demo_replay_001")
        use_replay = st.checkbox("Use replay (no model needed)", True)
        if st.button("Run Simulation"):
            with st.spinner("Running..."):
                try:
                    if use_replay:
                        from src.experimentation.simulate_campaign import run_replay
                        data_path = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
                        if not data_path.exists():
                            data_path = Path("data/raw/Telco-Customer-Churn.csv")
                        res = run_replay(
                            experiment_id=demo_exp_id,
                            data_path=str(data_path),
                            max_customers=1500,
                            data_dir=str(ROOT / "data" / "experiments"),
                        )
                    else:
                        from src.experimentation.simulate_campaign import run_from_model
                        res = run_from_model(
                            experiment_id=demo_exp_id,
                            data_dir=str(ROOT / "data" / "experiments"),
                        )
                    st.success(f"Done! Control: {res['n_control']}, Treatment: {res['n_treatment']}")
                    st.json(res)
                except Exception as e:
                    st.error(str(e))
                    import traceback
                    st.code(traceback.format_exc())
    
    with tab3:
        st.header("Results")
        exp_ids = get_experiment_ids()
        sel_exp = st.selectbox("Select experiment", ["---"] + exp_ids) if exp_ids else "---"
        
        if sel_exp and sel_exp != "---":
            if st.button("Run Analysis"):
                with st.spinner("Analyzing..."):
                    try:
                        from src.experimentation.analyze import run_analysis
                        result = run_analysis(
                            experiment_id=sel_exp,
                            data_dir=str(ROOT / "data" / "experiments"),
                            artifacts_dir=str(ROOT / "artifacts" / "experiments"),
                            segment_cols=["tenure", "Contract", "InternetService"],
                        )
                        st.success("Analysis complete!")
                    except Exception as e:
                        st.error(str(e))
        
        exp_name, analysis = load_latest_analysis()
        if analysis:
            st.subheader(f"Results: {exp_name}")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("SRM", "✓ Pass" if analysis.get("srm_passed") else "✗ Fail", delta=None)
                st.metric("Control N", analysis.get("control_stats", {}).get("n", 0))
                st.metric("Treatment N", analysis.get("treatment_stats", {}).get("n", 0))
            with c2:
                lift = analysis.get("lift")
                st.metric("Lift", f"{lift:.4f}" if lift is not None else "—")
                pv = analysis.get("p_value")
                st.metric("p-value", f"{pv:.4f}" if pv is not None else "—")
                st.metric("95% CI", f"[{analysis.get('ci_low', 0):.4f}, {analysis.get('ci_high', 0):.4f}]")
            
            if analysis.get("segments"):
                st.subheader("Segment Table")
                seg_df = analysis["segments"]
                st.dataframe(seg_df, use_container_width=True)
            
            lift_chart = ROOT / "artifacts" / "experiments" / exp_name / "lift_chart.png"
            if lift_chart.exists():
                st.image(str(lift_chart))
        else:
            st.info("Run a demo first, then run analysis to see results.")
    
    with tab4:
        st.header("Recommendation")
        exp_name, analysis = load_latest_analysis()
        if analysis:
            rec = analysis.get("recommendation", "iterate")
            reason = analysis.get("recommendation_reason", "")
            css = {"ship": "recommendation-ship", "hold": "recommendation-hold", "iterate": "recommendation-iterate"}
            st.markdown(f'<div class="{css.get(rec, "recommendation-iterate")}">**{rec.upper()}**: {reason}</div>', unsafe_allow_html=True)
            st.caption("Based on effect size, statistical significance, and SRM status.")
        else:
            st.info("Load an analysis to see the recommendation.")

if __name__ == "__main__":
    main()
