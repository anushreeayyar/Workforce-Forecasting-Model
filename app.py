"""
app.py — Workforce Forecasting Model
--------------------------------------
Interactive Streamlit dashboard answering 5 core workforce planning questions:
  1. How many people do we need to hire next quarter?
  2. What happens if attrition spikes?
  3. Where are our future capability gaps?
  4. Can we afford our growth plan?
  5. What's the right hiring velocity to hit our targets?

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os, sys

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from forecasting_model import (
    forecast_headcount,
    scenario_analysis,
    hiring_velocity,
    cost_projection,
    gap_analysis,
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Workforce Forecasting Model",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background-color: #0f1117; }
  .metric-card {
      background: linear-gradient(135deg, #1e2130, #252840);
      border: 1px solid #3d4270;
      border-radius: 12px;
      padding: 20px 24px;
      margin: 6px 0;
  }
  .metric-value {
      font-size: 2.2rem;
      font-weight: 700;
      color: #7c83fd;
  }
  .metric-label {
      font-size: 0.85rem;
      color: #8b8fa8;
      margin-top: 4px;
  }
  .metric-delta-pos { color: #4caf8e; font-size: 0.9rem; }
  .metric-delta-neg { color: #e05c5c; font-size: 0.9rem; }
  .section-header {
      font-size: 1.25rem;
      font-weight: 600;
      color: #c5c8e8;
      border-bottom: 2px solid #3d4270;
      padding-bottom: 8px;
      margin-bottom: 16px;
  }
  .insight-box {
      background: #1a1d2e;
      border-left: 4px solid #7c83fd;
      border-radius: 6px;
      padding: 14px 18px;
      font-size: 0.95rem;
      color: #c5c8e8;
      margin: 10px 0;
  }
  .critical-badge {
      background: #3d1515; color: #e05c5c;
      border-radius: 6px; padding: 3px 10px; font-size: 0.8rem;
  }
  .moderate-badge {
      background: #2d2a10; color: #f0c040;
      border-radius: 6px; padding: 3px 10px; font-size: 0.8rem;
  }
  .ok-badge {
      background: #0d2e1f; color: #4caf8e;
      border-radius: 6px; padding: 3px 10px; font-size: 0.8rem;
  }
  .stTabs [data-baseweb="tab"] {
      font-size: 0.95rem;
      font-weight: 500;
  }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    base = os.path.join(os.path.dirname(__file__), "data")
    hc   = pd.read_csv(f"{base}/headcount_snapshots.csv")
    emp  = pd.read_csv(f"{base}/employee_data.csv")
    ta   = pd.read_csv(f"{base}/ta_funnel.csv")
    return hc, emp, ta

try:
    hc_df, emp_df, ta_df = load_data()
except FileNotFoundError:
    st.error("⚠️  Data files not found. Run `python generate_synthetic_data.py` first.")
    st.stop()

DEPARTMENTS  = sorted(hc_df["department"].unique().tolist())
DEPT_COLORS  = {
    "Engineering":      "#7c83fd",
    "Sales":            "#fd7c7c",
    "Product":          "#7cfdcc",
    "HR":               "#fdb97c",
    "Finance":          "#c87cfd",
    "Operations":       "#7cc8fd",
    "Customer Success": "#fdec7c",
    "Data":             "#fd7cb9",
}

BUSINESS_UNITS = {
    "All Business Units":  DEPARTMENTS,
    "Technology":          ["Engineering", "Data", "Product"],
    "Commercial":          ["Sales", "Customer Success"],
    "Corporate":           ["HR", "Finance", "Operations"],
}

JOB_LEVELS = {
    "All Levels":              ["IC1","IC2","IC3","IC4","IC5","Manager","Director"],
    "Individual Contributor":  ["IC1","IC2","IC3","IC4","IC5"],
    "Manager":                 ["Manager"],
    "Director / VP (EMT)":     ["Director"],
}

EMT_LEVELS = ["Director"]

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 👥 Workforce Forecasting")
    st.markdown("*Synthetic data · Workforce planning methodology*")
    st.divider()

    st.markdown("### 🏢 Filters")

    # Business Unit
    selected_bu = st.selectbox(
        "Business Unit",
        list(BUSINESS_UNITS.keys()),
        index=0,
    )
    bu_departments = BUSINESS_UNITS[selected_bu]

    # EMT filter
    emt_only = st.toggle("EMT Only", value=False,
                         help="Show Director-level roles only")

    # Job Level
    selected_level_group = st.selectbox(
        "Job Level",
        list(JOB_LEVELS.keys()),
        index=0,
    )
    active_levels = JOB_LEVELS[selected_level_group]

    st.divider()
    st.markdown("### 🏢 Department")
    selected_dept = st.selectbox(
        "Select Department",
        sorted(bu_departments),
        index=0,
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("### 🎛️ Scenario Parameters")

    attrition_rate = st.slider(
        "Annual Attrition Rate",
        min_value=5, max_value=35, value=12, step=1,
        format="%d%%",
        help="Percentage of employees expected to leave annually"
    ) / 100

    growth_target = st.slider(
        "Headcount Growth Target",
        min_value=0, max_value=50, value=10, step=1,
        format="%d%%",
        help="Target headcount increase over the forecast horizon"
    ) / 100

    offer_accept_rate = st.slider(
        "Offer Acceptance Rate",
        min_value=40, max_value=95, value=75, step=1,
        format="%d%%",
        help="Percentage of job offers accepted by candidates"
    ) / 100

    horizon = st.selectbox(
        "Forecast Horizon",
        [6, 9, 12],
        index=2,
        format_func=lambda x: f"{x} months"
    )

    st.divider()
    st.markdown("### ℹ️ About")
    st.caption(
        "Built to demonstrate workforce planning methodology. "
        "Uses synthetic data generated with statistically realistic patterns "
        "matching enterprise HR environments."
    )

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
emt_label   = " · EMT Only" if emt_only else ""
level_label = f" · {selected_level_group}" if selected_level_group != "All Levels" else ""
st.markdown(f"# 👥 Workforce Forecasting Model")
st.markdown(f"**BU:** {selected_bu} &nbsp;·&nbsp; **Dept:** {selected_dept} &nbsp;·&nbsp; "
            f"**Level:** {selected_level_group}{emt_label} &nbsp;·&nbsp; "
            f"**Horizon:** {horizon}m &nbsp;·&nbsp; "
            f"**Attrition:** {attrition_rate*100:.0f}% &nbsp;·&nbsp; "
            f"**Growth:** {growth_target*100:.0f}% &nbsp;·&nbsp; "
            f"**Accept Rate:** {offer_accept_rate*100:.0f}%")
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE FORECASTS
# ══════════════════════════════════════════════════════════════════════════════
# Apply EMT / Job Level filter to employee data
filtered_emp = emp_df[emp_df["department"].isin(bu_departments)].copy()
if emt_only:
    filtered_emp = filtered_emp[filtered_emp["level"].isin(EMT_LEVELS)]
else:
    filtered_emp = filtered_emp[filtered_emp["level"].isin(active_levels)]

dept_hc      = hc_df[hc_df["department"] == selected_dept].sort_values("month")
current_hc   = dept_hc["headcount_actual"].iloc[-1]
target_hc    = dept_hc["headcount_target"].iloc[-1]

forecast_df  = forecast_headcount(hc_df, selected_dept, horizon)
scenario_df  = scenario_analysis(hc_df, selected_dept, attrition_rate, growth_target, horizon)
cost_df      = cost_projection(hc_df, scenario_df, selected_dept)
# Filter headcount data to selected Business Unit for gap analysis
bu_hc_df     = hc_df[hc_df["department"].isin(bu_departments)]
gaps_df      = gap_analysis(bu_hc_df, 6)
dept_ta      = ta_df[ta_df["department"] == selected_dept].sort_values("month")

total_hires_needed    = scenario_df["new_hires_needed"].sum()
scenario_end_hc       = scenario_df["headcount"].iloc[-1]
avg_ttf               = int(dept_ta["time_to_fill_days"].mean())
hv                    = hiring_velocity(total_hires_needed, horizon, offer_accept_rate, avg_ttf)
total_cost_12m        = cost_df["total_monthly_cost"].sum()
dept_gap_row          = gaps_df[gaps_df["department"] == selected_dept].iloc[0]

# ══════════════════════════════════════════════════════════════════════════════
# KPI STRIP
# ══════════════════════════════════════════════════════════════════════════════
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.metric("Current Headcount", f"{current_hc:,}", delta=f"Target: {target_hc:,}")
with k2:
    delta_hc = scenario_end_hc - current_hc
    st.metric(
        f"Projected HC ({horizon}m)",
        f"{scenario_end_hc:,}",
        delta=f"+{delta_hc:,} net adds" if delta_hc >= 0 else f"{delta_hc:,}",
        delta_color="normal"
    )
with k3:
    st.metric("Total Hires Needed", f"{total_hires_needed:,}",
              delta=f"{hv['offers_per_week']} offers/wk needed")
with k4:
    st.metric("12-Month Cost Impact", f"${total_cost_12m/1_000_000:.1f}M",
              delta=f"Avg ${cost_df['total_monthly_cost'].mean()/1000:.0f}K/mo")
with k5:
    gap_val = int(dept_gap_row["gap"])
    severity = dept_gap_row["gap_severity"]
    color = "inverse" if gap_val < 0 else "off"
    st.metric("6-Month HC Gap", f"{gap_val:+,}", delta=severity, delta_color=color)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Headcount Forecast",
    "🎛️ Scenario Analysis",
    "🏢 Gap Analysis",
    "💰 Cost Projection",
    "🎯 Hiring Velocity",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — HEADCOUNT FORECAST
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown(f'<div class="section-header">📈 {selected_dept} — 12-Month Headcount Forecast</div>',
                unsafe_allow_html=True)

    col_chart, col_insight = st.columns([3, 1])

    with col_chart:
        fig = go.Figure()

        # Historical actual
        fig.add_trace(go.Scatter(
            x=dept_hc["month"], y=dept_hc["headcount_actual"],
            name="Historical Actual",
            line=dict(color=DEPT_COLORS.get(selected_dept, "#7c83fd"), width=2.5),
            mode="lines+markers", marker=dict(size=4),
        ))
        # Historical target
        fig.add_trace(go.Scatter(
            x=dept_hc["month"], y=dept_hc["headcount_target"],
            name="Headcount Target",
            line=dict(color="#8b8fa8", width=1.5, dash="dash"),
        ))
        # 95% CI band
        fig.add_trace(go.Scatter(
            x=forecast_df["month"].tolist() + forecast_df["month"].tolist()[::-1],
            y=forecast_df["upper_95"].tolist() + forecast_df["lower_95"].tolist()[::-1],
            fill="toself",
            fillcolor="rgba(124,131,253,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% Confidence",
            showlegend=True,
        ))
        # 80% CI band
        fig.add_trace(go.Scatter(
            x=forecast_df["month"].tolist() + forecast_df["month"].tolist()[::-1],
            y=forecast_df["upper_80"].tolist() + forecast_df["lower_80"].tolist()[::-1],
            fill="toself",
            fillcolor="rgba(124,131,253,0.18)",
            line=dict(color="rgba(0,0,0,0)"),
            name="80% Confidence",
            showlegend=True,
        ))
        # Point forecast
        fig.add_trace(go.Scatter(
            x=forecast_df["month"], y=forecast_df["forecast"],
            name="Forecast",
            line=dict(color=DEPT_COLORS.get(selected_dept, "#7c83fd"), width=2.5, dash="dot"),
            mode="lines+markers", marker=dict(size=5, symbol="diamond"),
        ))
        # Vertical line at forecast start — scatter trace works on all Plotly versions
        last_hist = dept_hc["month"].iloc[-1]
        y_min = min(dept_hc["headcount_actual"].min(), forecast_df["lower_95"].min()) * 0.95
        y_max = max(dept_hc["headcount_actual"].max(), forecast_df["upper_95"].max()) * 1.05
        fig.add_trace(go.Scatter(
            x=[last_hist, last_hist], y=[y_min, y_max],
            mode="lines", line=dict(dash="dash", color="#4a4e6a", width=1.5),
            showlegend=False, hoverinfo="skip", name="",
        ))

        fig.update_layout(
            template="plotly_dark",
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            xaxis=dict(title="Month", gridcolor="#2a2d3e"),
            yaxis=dict(title="Headcount", gridcolor="#2a2d3e"),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_insight:
        end_forecast = forecast_df["forecast"].iloc[-1]
        q1_end       = forecast_df["forecast"].iloc[2] if len(forecast_df) >= 3 else end_forecast
        q1_hires_est = scenario_df["new_hires_needed"].iloc[:3].sum()

        st.markdown('<div class="section-header">Key Insights</div>', unsafe_allow_html=True)
        st.markdown(f"""
<div class="insight-box">
📌 <b>End-of-horizon forecast:</b><br>{end_forecast:,} headcount<br>
<span style="color:#8b8fa8;font-size:0.85rem">
  Range: {forecast_df["lower_95"].iloc[-1]:,} – {forecast_df["upper_95"].iloc[-1]:,} (95% CI)
</span>
</div>
<div class="insight-box">
📅 <b>Next Quarter (3 months):</b><br>~{q1_hires_est:,} hires needed
</div>
<div class="insight-box">
📉 <b>Historical avg attrition:</b><br>{dept_hc["attritions"].mean():.1f} / month
</div>
<div class="insight-box">
📈 <b>Historical avg hires:</b><br>{dept_hc["new_hires"].mean():.1f} / month
</div>
""", unsafe_allow_html=True)

    # Historical trend table
    with st.expander("📋 View Historical Data"):
        st.dataframe(
            dept_hc[["month","headcount_actual","headcount_target","new_hires","attritions","open_reqs"]]
            .tail(12).sort_values("month", ascending=False),
            use_container_width=True, hide_index=True
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — SCENARIO ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown(f'<div class="section-header">🎛️ Scenario: {attrition_rate*100:.0f}% Attrition · {growth_target*100:.0f}% Growth</div>',
                unsafe_allow_html=True)

    # Live insight banner
    st.markdown(f"""
<div class="insight-box" style="font-size:1.05rem;">
🔮 <b>At {attrition_rate*100:.0f}% attrition and {growth_target*100:.0f}% growth target,
{selected_dept} needs <span style="color:#7c83fd">{total_hires_needed:,} total hires</span>
over the next {horizon} months — reaching a headcount of {scenario_end_hc:,} by month {horizon}.</b>
</div>
""", unsafe_allow_html=True)

    # Baseline vs scenario comparison
    col_a, col_b = st.columns(2)

    with col_a:
        # Headcount trajectory
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=scenario_df["month"],
            y=scenario_df["attritions"],
            name="Monthly Attritions",
            marker_color="#e05c5c",
            opacity=0.7,
        ))
        fig2.add_trace(go.Bar(
            x=scenario_df["month"],
            y=scenario_df["new_hires_needed"],
            name="Hires Needed",
            marker_color="#4caf8e",
            opacity=0.7,
        ))
        fig2.add_trace(go.Scatter(
            x=scenario_df["month"],
            y=scenario_df["headcount"],
            name="Projected Headcount",
            yaxis="y2",
            line=dict(color="#7c83fd", width=2.5),
            mode="lines+markers",
        ))
        fig2.update_layout(
            template="plotly_dark",
            title="Attritions vs Hires Needed (monthly)",
            barmode="group",
            height=360,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            yaxis=dict(title="People", gridcolor="#2a2d3e"),
            yaxis2=dict(title="Headcount", overlaying="y", side="right",
                        gridcolor="rgba(0,0,0,0)"),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        # Scenario comparison table
        scenarios = []
        for a, g, label in [
            (0.08, 0.05, "Optimistic"),
            (attrition_rate, growth_target, "Your Scenario ⬅"),
            (0.20, 0.15, "Moderate Stress"),
            (0.30, 0.20, "High Stress"),
        ]:
            sc = scenario_analysis(hc_df, selected_dept, a, g, horizon)
            scenarios.append({
                "Scenario":        label,
                "Attrition":       f"{a*100:.0f}%",
                "Growth Target":   f"{g*100:.0f}%",
                "Total Hires":     sc["new_hires_needed"].sum(),
                "End Headcount":   sc["headcount"].iloc[-1],
                "Net Change":      sc["headcount"].iloc[-1] - current_hc,
            })
        sc_df = pd.DataFrame(scenarios)

        st.markdown("#### Scenario Comparison")
        st.dataframe(sc_df, use_container_width=True, hide_index=True)

        # Cumulative hires chart
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=scenario_df["month"],
            y=scenario_df["new_hires_needed"].cumsum(),
            fill="tozeroy",
            name="Cumulative Hires",
            line=dict(color="#4caf8e", width=2),
            fillcolor="rgba(76,175,142,0.15)",
        ))
        fig3.update_layout(
            template="plotly_dark",
            title="Cumulative Hires Required",
            height=260,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="Cumulative Hires", gridcolor="#2a2d3e"),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Detail table
    with st.expander("📋 Month-by-Month Scenario Detail"):
        st.dataframe(scenario_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — GAP ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">🏢 6-Month Headcount Gap — All Departments</div>',
                unsafe_allow_html=True)

    col_gap1, col_gap2 = st.columns([3, 2])

    with col_gap1:
        fig_gap = go.Figure()
        colors  = ["#e05c5c" if g < -5 else "#f0c040" if g < 0 else "#4caf8e"
                   for g in gaps_df["gap"]]
        fig_gap.add_trace(go.Bar(
            x=gaps_df["department"],
            y=gaps_df["gap"],
            marker_color=colors,
            text=[f"{g:+d}" for g in gaps_df["gap"]],
            textposition="outside",
        ))
        fig_gap.add_hline(y=0, line_color="#4a4e6a", line_dash="dash")
        fig_gap.update_layout(
            template="plotly_dark",
            title="Projected Headcount Gap at 6 Months (Projected − Target)",
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="Gap (people)", gridcolor="#2a2d3e"),
            xaxis=dict(title="", gridcolor="#2a2d3e"),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_gap, use_container_width=True)

    with col_gap2:
        st.markdown("#### Severity Breakdown")
        for _, row in gaps_df.iterrows():
            badge = (
                '<span class="critical-badge">CRITICAL</span>'  if row["gap_severity"] == "Critical"
                else '<span class="moderate-badge">MODERATE</span>' if row["gap_severity"] == "Moderate"
                else '<span class="ok-badge">ON TRACK</span>'
            )
            st.markdown(
                f"""<div class="insight-box" style="margin:4px 0; padding:10px 14px;">
                {badge} &nbsp;<b>{row["department"]}</b><br>
                <span style="color:#8b8fa8;font-size:0.82rem;">
                  Current: {row["current_headcount"]} &nbsp;|&nbsp;
                  Target: {row["target_headcount"]} &nbsp;|&nbsp;
                  Gap: {row["gap"]:+d}
                </span></div>""",
                unsafe_allow_html=True,
            )

    # Radar chart — headcount utilization
    dept_names   = gaps_df["department"].tolist()
    utilizations = [
        min(100, round(row["projected_headcount"] / max(row["target_headcount"], 1) * 100))
        for _, row in gaps_df.iterrows()
    ]
    fig_radar = go.Figure(go.Scatterpolar(
        r=utilizations + [utilizations[0]],
        theta=dept_names + [dept_names[0]],
        fill="toself",
        fillcolor="rgba(124,131,253,0.2)",
        line=dict(color="#7c83fd"),
        name="HC Utilization %",
    ))
    fig_radar.update_layout(
        template="plotly_dark",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 130], gridcolor="#2a2d3e"),
            angularaxis=dict(gridcolor="#2a2d3e"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        title="Headcount Utilization (Projected / Target %)",
        height=350,
        margin=dict(l=30, r=30, t=50, b=30),
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — COST PROJECTION
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown(f'<div class="section-header">💰 {selected_dept} — 12-Month Cost Projection</div>',
                unsafe_allow_html=True)

    total_salary   = cost_df["salary_cost"].sum()
    total_benefits = cost_df["benefits_cost"].sum()
    total_recruit  = cost_df["recruiting_cost"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total 12-Month Cost", f"${total_cost_12m/1_000_000:.2f}M")
    c2.metric("Salary Spend",         f"${total_salary/1_000_000:.2f}M")
    c3.metric("Benefits Spend",       f"${total_benefits/1_000_000:.2f}M")
    c4.metric("Recruiting Spend",     f"${total_recruit/1_000:.0f}K")

    # Stacked area chart
    fig_cost = go.Figure()
    fig_cost.add_trace(go.Scatter(
        x=cost_df["month"], y=cost_df["salary_cost"],
        name="Salary", stackgroup="one",
        fillcolor="rgba(124,131,253,0.7)", line=dict(color="rgba(0,0,0,0)"),
    ))
    fig_cost.add_trace(go.Scatter(
        x=cost_df["month"], y=cost_df["benefits_cost"],
        name="Benefits", stackgroup="one",
        fillcolor="rgba(76,175,142,0.7)", line=dict(color="rgba(0,0,0,0)"),
    ))
    fig_cost.add_trace(go.Scatter(
        x=cost_df["month"], y=cost_df["recruiting_cost"],
        name="Recruiting", stackgroup="one",
        fillcolor="rgba(240,192,64,0.7)", line=dict(color="rgba(0,0,0,0)"),
    ))
    fig_cost.add_trace(go.Scatter(
        x=cost_df["month"], y=cost_df["cumulative_cost"],
        name="Cumulative Cost",
        yaxis="y2",
        line=dict(color="#fd7c7c", width=2.5, dash="dot"),
    ))
    fig_cost.update_layout(
        template="plotly_dark",
        title="Monthly Cost Breakdown + Cumulative Spend",
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis=dict(title="Monthly Cost ($)", gridcolor="#2a2d3e",
                   tickformat="$,.0f"),
        yaxis2=dict(title="Cumulative ($)", overlaying="y", side="right",
                    gridcolor="rgba(0,0,0,0)", tickformat="$,.0f"),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig_cost, use_container_width=True)

    # Cross-dept cost comparison
    st.markdown("#### Cost Comparison Across Departments")
    dept_costs = []
    for dept in sorted(bu_departments):
        sc_d  = scenario_analysis(hc_df, dept, attrition_rate, growth_target, 12)
        cp_d  = cost_projection(hc_df, sc_d, dept)
        dept_costs.append({
            "Department": dept,
            "Total Cost":         f"${cp_d['total_monthly_cost'].sum()/1_000_000:.2f}M",
            "Avg Monthly":        f"${cp_d['total_monthly_cost'].mean()/1_000:.0f}K",
            "Total Hires":        sc_d["new_hires_needed"].sum(),
            "Recruiting Spend":   f"${cp_d['recruiting_cost'].sum()/1_000:.0f}K",
            "End Headcount":      sc_d["headcount"].iloc[-1],
        })
    st.dataframe(pd.DataFrame(dept_costs), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — HIRING VELOCITY
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown(f'<div class="section-header">🎯 Hiring Velocity — {selected_dept}</div>',
                unsafe_allow_html=True)

    # Main answer
    st.markdown(f"""
<div class="insight-box" style="font-size:1.1rem; padding:20px 24px;">
🎯 <b>To hire {total_hires_needed:,} people in {horizon} months at a {offer_accept_rate*100:.0f}% offer
acceptance rate, the {selected_dept} TA team needs to send
<span style="color:#7c83fd; font-size:1.4rem;">{hv['offers_per_week']} offers per week</span>
and maintain a pipeline of
<span style="color:#4caf8e;">{hv['pipeline_needed']:,} active candidates</span> at all times.</b>
</div>
""", unsafe_allow_html=True)

    v1, v2, v3, v4, v5 = st.columns(5)
    v1.metric("Total Hires Needed",   f"{hv['total_hires_needed']:,}")
    v2.metric("Total Offers Required", f"{hv['total_offers_needed']:,}")
    v3.metric("Offers / Week",         f"{hv['offers_per_week']}")
    v4.metric("Hires / Month",         f"{hv['hires_per_month']}")
    v5.metric("Pipeline Size Needed",  f"{hv['pipeline_needed']:,}")

    col_v1, col_v2 = st.columns(2)

    with col_v1:
        # Offers per week sensitivity to acceptance rate
        accept_rates  = np.arange(0.40, 0.96, 0.05)
        opw_values    = [
            hiring_velocity(total_hires_needed, horizon, ar, avg_ttf)["offers_per_week"]
            for ar in accept_rates
        ]
        fig_hv = go.Figure()
        fig_hv.add_trace(go.Scatter(
            x=[f"{r*100:.0f}%" for r in accept_rates],
            y=opw_values,
            mode="lines+markers",
            line=dict(color="#7c83fd", width=2.5),
            marker=dict(size=7),
            fill="tozeroy",
            fillcolor="rgba(124,131,253,0.15)",
            name="Offers/Week",
        ))
        # Highlight current rate — scatter trace works on all Plotly versions
        current_idx = int((offer_accept_rate - 0.40) / 0.05)
        if 0 <= current_idx < len(accept_rates):
            cur_x = f"{accept_rates[current_idx]*100:.0f}%"
            fig_hv.add_trace(go.Scatter(
                x=[cur_x, cur_x], y=[0, max(opw_values) * 1.1],
                mode="lines", line=dict(dash="dash", color="#fd7c7c", width=1.5),
                showlegend=False, hoverinfo="skip", name="",
            ))
        fig_hv.update_layout(
            template="plotly_dark",
            title="Offers/Week Sensitivity to Accept Rate",
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="Offers per Week", gridcolor="#2a2d3e"),
            xaxis=dict(title="Offer Acceptance Rate"),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_hv, use_container_width=True)

    with col_v2:
        # TA funnel trend
        dept_ta_recent = dept_ta.tail(12)
        fig_funnel = go.Figure()
        for col_name, color, label in [
            ("applications",    "#8b8fa8", "Applications"),
            ("interviews",      "#7cc8fd", "Interviews"),
            ("offers_made",     "#fdb97c", "Offers Made"),
            ("offers_accepted", "#4caf8e", "Offers Accepted"),
        ]:
            fig_funnel.add_trace(go.Scatter(
                x=dept_ta_recent["month"],
                y=dept_ta_recent[col_name],
                name=label,
                mode="lines+markers",
                line=dict(color=color, width=2),
            ))
        fig_funnel.update_layout(
            template="plotly_dark",
            title="Historical TA Funnel (Last 12 Months)",
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            yaxis=dict(title="Count", gridcolor="#2a2d3e"),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_funnel, use_container_width=True)

    # Time-to-fill trend
    fig_ttf = go.Figure()
    fig_ttf.add_trace(go.Bar(
        x=dept_ta_recent["month"],
        y=dept_ta_recent["time_to_fill_days"],
        marker_color=[
            "#e05c5c" if v > 40 else "#f0c040" if v > 30 else "#4caf8e"
            for v in dept_ta_recent["time_to_fill_days"]
        ],
        name="Time to Fill (days)",
        text=dept_ta_recent["time_to_fill_days"].apply(lambda x: f"{x}d"),
        textposition="outside",
    ))
    fig_ttf.add_hline(y=avg_ttf, line_dash="dash", line_color="#7c83fd",
                      annotation_text=f"Dept avg: {avg_ttf}d")
    fig_ttf.update_layout(
        template="plotly_dark",
        title="Time-to-Fill Trend (days)",
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="Days", gridcolor="#2a2d3e"),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig_ttf, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown(
    "<div style='text-align:center; color:#4a4e6a; font-size:0.8rem;'>"
    "Workforce Forecasting Model · Built with Streamlit + Plotly · "
    "Synthetic data generated with statistically realistic HR patterns · "
    "Methodology mirrors enterprise workforce planning frameworks"
    "</div>",
    unsafe_allow_html=True,
)
