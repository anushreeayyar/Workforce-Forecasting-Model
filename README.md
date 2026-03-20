# 👥 Workforce Forecasting Model

An interactive workforce planning tool built with **Streamlit + Plotly** that answers the five core questions every CHRO, CFO, and VP of People needs answered:

> *"How many people will we need, in which teams, and when — and what happens if our attrition or hiring plans change?"*

---

## 🎯 The Five Questions This Tool Answers

| # | Question | Business Value |
|---|----------|---------------|
| 1 | How many people do we need to hire next quarter? | TA teams plan recruiter capacity and budget ahead of time |
| 2 | What happens if attrition spikes? | Stress-test your workforce plan before it becomes a crisis |
| 3 | Where are our future capability gaps? | L&D plans training before the gap hits, not after |
| 4 | Can we afford our growth plan? | HR + Finance align on a number before headcount goes to the board |
| 5 | What's the right hiring velocity to hit our targets? | TA teams don't miss targets because the math wasn't done upfront |

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/workforce-forecasting-model.git
cd workforce-forecasting-model
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate the synthetic dataset
```bash
python generate_synthetic_data.py
```
This creates three CSV files in `/data`:
- `headcount_snapshots.csv` — 36 months of monthly headcount by department
- `employee_data.csv` — individual employee records (synthetic, anonymized)
- `ta_funnel.csv` — monthly recruiting pipeline by department

### 4. Launch the app
```bash
streamlit run app.py
```

---

## 📊 Dashboard Overview

The app has five interactive tabs:

**📈 Headcount Forecast**
- 12-month headcount projection with 80% and 95% confidence intervals
- Monte Carlo simulation (500 runs) for uncertainty quantification
- Historical trend + target overlay

**🎛️ Scenario Analysis**
- Adjust attrition rate (5–35%), growth target (0–50%), and offer acceptance rate (40–95%) via sliders
- Live update: "At 20% attrition and 15% growth, Engineering needs X hires by Q3"
- Side-by-side comparison of 4 stress-test scenarios (Optimistic / Your Scenario / Moderate Stress / High Stress)

**🏢 Gap Analysis**
- Heatmap of projected headcount gaps across all 8 departments
- Severity classification: Critical / Moderate / On Track
- Radar chart of headcount utilization

**💰 Cost Projection**
- Stacked monthly cost (salary + benefits + recruiting) with cumulative spend line
- Cross-department cost comparison table
- Adjusts dynamically to scenario sliders

**🎯 Hiring Velocity**
- Exact offers-per-week calculation needed to hit hiring targets
- Sensitivity chart: how acceptance rate affects required offer volume
- Historical TA funnel visualization + time-to-fill trend

---

## 🗂️ Project Structure

```
workforce-forecasting-model/
├── app.py                        # Streamlit dashboard (main entry point)
├── forecasting_model.py          # Core forecasting logic (no sklearn needed)
├── generate_synthetic_data.py    # Synthetic HR data generator
├── requirements.txt
├── .gitignore
├── README.md
└── data/
    ├── headcount_snapshots.csv   # Monthly headcount by dept (288 rows)
    ├── employee_data.csv         # Employee records (1,065 rows)
    └── ta_funnel.csv             # Recruiting pipeline (288 rows)
```

---

## 🧠 Methodology

### Forecasting
- **Trend extrapolation**: OLS linear regression on historical headcount (pure NumPy, no sklearn)
- **Confidence intervals**: Monte Carlo simulation (500 paths) with residual-based noise + horizon drift
- **Seasonality**: Monthly hiring factors derived from typical enterprise hiring patterns (Q1/Q3 peaks)

### Scenario Modeling
- Month-by-month simulation of headcount evolution
- Attrition applied monthly (`annual_rate / 12 × current_headcount`)
- Growth modeled as a linear ramp toward the target

### Hiring Velocity
- `offers_needed = hires_needed / offer_accept_rate`
- Pipeline size accounts for time-to-fill days

### Cost Model
- `Total cost = salary × 1.25 (benefits) + new_hires × salary × 0.15 (recruiting)`

---

## 📁 Dataset Description

All data is **synthetically generated** using statistically realistic patterns:

| Dataset | Rows | Key Fields |
|---------|------|-----------|
| headcount_snapshots | 288 | month, department, headcount_actual, headcount_target, new_hires, attritions, open_reqs |
| employee_data | ~1,065 | employee_id, department, level, hire_date, exit_date, exit_reason, annual_salary |
| ta_funnel | 288 | month, department, roles_open, applications, interviews, offers_made, offers_accepted, time_to_fill_days |

Departments: Engineering, Sales, Product, HR, Finance, Operations, Customer Success, Data

---

## 🔧 Extending This Project

- **Plug in real data**: Replace `/data/*.csv` with your actual (anonymized) exports from Workday, BambooHR, Greenhouse, etc.
- **Add ML forecasting**: Swap `_LinearRegression` in `forecasting_model.py` with Prophet or ARIMA for time-series-aware forecasting
- **Add skills taxonomy**: Extend `employee_data.csv` with skills/role columns to power capability gap analysis
- **Connect to ATS**: Use Greenhouse or Lever APIs to pull live TA funnel data

---

## 📝 Note on Data

This project uses **100% synthetic data** generated by `generate_synthetic_data.py`. It demonstrates the same methodology applied to real workforce planning — no proprietary data is included.

---

*Built to demonstrate enterprise-grade workforce planning methodology. Methodology mirrors scenario-based forecasting frameworks used in large-scale HR organizations.*
