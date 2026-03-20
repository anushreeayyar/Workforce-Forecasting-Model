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
| 4 | Can we afford our growth plan? | HR + Finance align before headcount goes to the board |
| 5 | What's the right hiring velocity to hit our targets? | TA teams don't miss targets because the math wasn't done upfront |

---

## 📊 Dashboard Overview

The app has five interactive tabs, each tied to one of the core questions above:

**📈 Headcount Forecast** — 12-month projection with 80% and 95% Monte Carlo confidence bands, historical trend overlay, and target tracking.

**🎛️ Scenario Analysis** — Adjust attrition rate, growth target, and offer acceptance rate via live sliders. Outputs a narrative: *"At 20% attrition and 15% growth, Engineering needs 23 additional hires by Q3."* Side-by-side comparison of four stress-test scenarios.

**🏢 Gap Analysis** — Projected headcount gaps across all departments with Critical / Moderate / On Track severity ratings and a radar chart of headcount utilization by Business Unit.

**💰 Cost Projection** — Stacked monthly cost breakdown (salary + benefits + recruiting) with cumulative spend line and cross-department comparison table.

**🎯 Hiring Velocity** — Exact offers-per-week calculation to hit hiring targets, sensitivity analysis on offer acceptance rate, and historical TA funnel visualization.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `pandas` / `numpy` | Data engineering & forecasting |
| `streamlit` | Interactive dashboard |
| `plotly` | Charts & visualisations |
| `numpy` (OLS) | Linear regression — no sklearn needed |
| Monte Carlo | Confidence interval simulation (500 paths) |

---

## 🚀 Quick Start

```bash
git clone https://github.com/anushreeayyar/Workforce-Forecasting-Model.git
cd Workforce-Forecasting-Model
pip install -r requirements.txt
python generate_synthetic_data.py
streamlit run app.py
```

---

## 🗂️ Project Structure

```
Workforce-Forecasting-Model/
├── app.py                        # Streamlit dashboard
├── forecasting_model.py          # Core forecasting logic
├── generate_synthetic_data.py    # HR data generator
├── requirements.txt
├── .gitignore
└── data/
    ├── headcount_snapshots.csv   # Monthly headcount by dept (288 rows)
    ├── employee_data.csv         # Employee records (1,065 rows)
    └── ta_funnel.csv             # Recruiting pipeline (288 rows)
```

---

## 🧠 Methodology

**Forecasting** — OLS linear regression on historical headcount with Monte Carlo simulation (500 paths) for confidence intervals. Seasonality factors applied for Q1/Q3 hiring peaks.

**Scenario Modeling** — Month-by-month simulation of headcount under custom attrition and growth targets.

**Hiring Velocity** — `offers_needed = hires_needed / offer_accept_rate`, with pipeline sizing based on time-to-fill.

**Cost Model** — `Total cost = salary × 1.25 (benefits) + new_hires × salary × 0.15 (recruiting)`

---

## 🔧 Filters & Controls

- **Business Unit** — Technology / Commercial / Corporate / All
- **Job Level** — All Levels / Individual Contributor / Manager / Director & VP
- **EMT Only** — toggle to view Director-level workforce planning
- **Scenario sliders** — attrition rate, growth target, offer acceptance rate, forecast horizon

---

## 🗺️ Roadmap

- [ ] Live HRIS integration (Workday / BambooHR API)
- [ ] Skills taxonomy layer for capability gap analysis
- [ ] Prophet / ARIMA time-series forecasting
- [ ] Streamlit Cloud deployment
- [ ] Automated monthly headcount report (PDF export)

---

## 👤 Author

Built by **Anushree Ayyar** — HR Tech / People Analytics portfolio project.

Connect on [LinkedIn](https://linkedin.com/in/anushreeayyar) · [GitHub](https://github.com/anushreeayyar)

---

## 📄 Licence

MIT — free to use, modify, and distribute.
