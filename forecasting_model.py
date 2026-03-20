"""
forecasting_model.py
---------------------
Core forecasting logic for the Workforce Forecasting Model.
Provides three forecasting functions used by the Streamlit app:

  - forecast_headcount()   → 12-month headcount projection with confidence bands
  - scenario_analysis()    → "what-if" headcount under custom attrition / growth
  - hiring_velocity()      → offers-per-week needed to hit headcount target
  - cost_projection()      → total salary cost under forecast scenario

All functions work on pandas DataFrames loaded from /data/*.csv
"""

import numpy as np
import pandas as pd


# ── Pure-NumPy linear regression (no sklearn dependency) ─────────────────────
class _LinearRegression:
    """Minimal OLS linear regression using numpy least-squares."""
    def fit(self, X, y):
        X_b = np.column_stack([np.ones(len(X)), X.flatten()])
        self.coef_ = np.linalg.lstsq(X_b, y, rcond=None)[0]
        return self

    def predict(self, X):
        X_b = np.column_stack([np.ones(len(X)), X.flatten()])
        return X_b @ self.coef_


# ── Constants ─────────────────────────────────────────────────────────────────
BENEFITS_MULTIPLIER   = 1.25   # benefits add ~25% on top of salary
RECRUITING_COST_RATIO = 0.15   # recruiting spend ≈ 15% of first-year salary
WEEKS_PER_MONTH       = 4.33


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _month_index(month_str: str) -> int:
    """Convert '2022-01' → integer for regression."""
    y, m = month_str.split("-")
    return int(y) * 12 + int(m)


def _add_months(month_str: str, n: int) -> str:
    y, m = map(int, month_str.split("-"))
    total  = (m - 1) + n          # 0-based months offset
    ny     = y + total // 12
    nm     = total % 12 + 1       # back to 1-based
    return f"{ny}-{nm:02d}"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. BASE FORECAST — trend + seasonality + confidence bands (Monte Carlo)
# ═══════════════════════════════════════════════════════════════════════════════

def forecast_headcount(
    hc_df: pd.DataFrame,
    department: str,
    horizon_months: int = 12,
    n_simulations: int = 500,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        month, forecast, lower_80, upper_80, lower_95, upper_95
    """
    dept_df = hc_df[hc_df["department"] == department].copy()
    dept_df["t"] = dept_df["month"].apply(_month_index)
    dept_df = dept_df.sort_values("t")

    X = dept_df["t"].values.reshape(-1, 1)
    y = dept_df["headcount_actual"].values.astype(float)

    reg = _LinearRegression().fit(X, y)
    residuals = y - reg.predict(X)
    resid_std = residuals.std()

    last_t     = dept_df["t"].iloc[-1]
    last_month = dept_df["month"].iloc[-1]

    future_ts      = np.array([last_t + i for i in range(1, horizon_months + 1)])
    point_forecast = reg.predict(future_ts.reshape(-1, 1))

    # Monte Carlo for uncertainty bands
    sim_matrix = np.zeros((n_simulations, horizon_months))
    for s in range(n_simulations):
        noise = np.random.normal(0, resid_std, horizon_months)
        # uncertainty grows with horizon
        drift = np.cumsum(np.random.normal(0, resid_std * 0.05, horizon_months))
        sim_matrix[s] = point_forecast + noise + drift

    sim_matrix = np.maximum(sim_matrix, 0)

    months_list = [_add_months(last_month, i) for i in range(1, horizon_months + 1)]

    return pd.DataFrame({
        "month":     months_list,
        "forecast":  np.round(point_forecast).astype(int),
        "lower_80":  np.round(np.percentile(sim_matrix, 10, axis=0)).astype(int),
        "upper_80":  np.round(np.percentile(sim_matrix, 90, axis=0)).astype(int),
        "lower_95":  np.round(np.percentile(sim_matrix, 2.5, axis=0)).astype(int),
        "upper_95":  np.round(np.percentile(sim_matrix, 97.5, axis=0)).astype(int),
    })


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SCENARIO ANALYSIS — what-if sliders
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_analysis(
    hc_df: pd.DataFrame,
    department: str,
    attrition_rate: float,       # e.g. 0.15 for 15% annual
    growth_target_pct: float,    # e.g. 0.10 for 10% headcount growth
    horizon_months: int = 12,
) -> pd.DataFrame:
    """
    Simulates month-by-month headcount evolution under custom attrition + growth.
    Returns columns: month, headcount, attritions, new_hires_needed
    """
    dept_df = hc_df[hc_df["department"] == department].copy()
    dept_df = dept_df.sort_values("month")

    current_hc   = dept_df["headcount_actual"].iloc[-1]
    last_month   = dept_df["month"].iloc[-1]
    target_end   = int(current_hc * (1 + growth_target_pct))

    monthly_attrition = attrition_rate / 12
    monthly_growth    = (target_end - current_hc) / horizon_months

    rows = []
    hc = current_hc
    for i in range(1, horizon_months + 1):
        month_str        = _add_months(last_month, i)
        attritions       = int(hc * monthly_attrition)
        hires_for_growth = int(monthly_growth)
        new_hires_needed = attritions + hires_for_growth
        hc               = max(0, hc - attritions + new_hires_needed)
        rows.append({
            "month":             month_str,
            "headcount":         hc,
            "attritions":        attritions,
            "new_hires_needed":  new_hires_needed,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. HIRING VELOCITY — offers per week
# ═══════════════════════════════════════════════════════════════════════════════

def hiring_velocity(
    total_hires_needed: int,
    months_available: int,
    offer_accept_rate: float,    # e.g. 0.75 for 75%
    time_to_fill_days: int = 35,
) -> dict:
    """
    Returns a dict with:
        offers_per_week, total_offers_needed, hires_per_month, pipeline_needed
    """
    total_offers_needed = int(np.ceil(total_hires_needed / max(offer_accept_rate, 0.01)))
    weeks_available     = months_available * WEEKS_PER_MONTH
    offers_per_week     = round(total_offers_needed / max(weeks_available, 1), 1)
    hires_per_month     = round(total_hires_needed / max(months_available, 1), 1)

    # How many active candidates need to be in pipeline at any given time
    pipeline_needed = int(np.ceil(
        offers_per_week * (time_to_fill_days / 7) / max(offer_accept_rate, 0.01)
    ))

    return {
        "total_hires_needed":   total_hires_needed,
        "total_offers_needed":  total_offers_needed,
        "months_available":     months_available,
        "offers_per_week":      offers_per_week,
        "hires_per_month":      hires_per_month,
        "pipeline_needed":      pipeline_needed,
        "offer_accept_rate":    offer_accept_rate,
        "time_to_fill_days":    time_to_fill_days,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. COST PROJECTION
# ═══════════════════════════════════════════════════════════════════════════════

def cost_projection(
    hc_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    department: str,
) -> pd.DataFrame:
    """
    Calculates monthly and cumulative cost of headcount under a given scenario.
    Includes salary cost, benefits, and recruiting cost for net-new hires.
    """
    dept_hc = hc_df[hc_df["department"] == department].copy()
    avg_salary = dept_hc["avg_salary"].mean()

    rows = []
    for _, row in scenario_df.iterrows():
        monthly_salary_cost    = row["headcount"] * (avg_salary / 12)
        monthly_benefits_cost  = monthly_salary_cost * (BENEFITS_MULTIPLIER - 1)
        recruiting_cost        = row["new_hires_needed"] * avg_salary * RECRUITING_COST_RATIO
        total_monthly_cost     = monthly_salary_cost + monthly_benefits_cost + recruiting_cost
        rows.append({
            "month":               row["month"],
            "headcount":           row["headcount"],
            "salary_cost":         round(monthly_salary_cost),
            "benefits_cost":       round(monthly_benefits_cost),
            "recruiting_cost":     round(recruiting_cost),
            "total_monthly_cost":  round(total_monthly_cost),
        })

    df = pd.DataFrame(rows)
    df["cumulative_cost"] = df["total_monthly_cost"].cumsum()
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 5. GAP ANALYSIS — which departments are most understaffed?
# ═══════════════════════════════════════════════════════════════════════════════

def gap_analysis(hc_df: pd.DataFrame, horizon_months: int = 6) -> pd.DataFrame:
    """
    Returns a summary of projected headcount gaps across all departments
    at `horizon_months` from the last data point.
    """
    rows = []
    for dept in hc_df["department"].unique():
        dept_df   = hc_df[hc_df["department"] == dept].sort_values("month")
        current   = dept_df["headcount_actual"].iloc[-1]
        target    = dept_df["headcount_target"].iloc[-1]
        forecast  = forecast_headcount(hc_df, dept, horizon_months)
        projected = forecast["forecast"].iloc[-1]
        gap       = projected - target
        rows.append({
            "department":            dept,
            "current_headcount":     current,
            "target_headcount":      target,
            "projected_headcount":   projected,
            "gap":                   gap,
            "gap_severity":          "Critical" if gap < -10 else "Moderate" if gap < -3 else "On Track",
        })

    return pd.DataFrame(rows).sort_values("gap")


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    hc = pd.read_csv("data/headcount_snapshots.csv")
    print("=== Headcount Forecast: Engineering ===")
    fc = forecast_headcount(hc, "Engineering")
    print(fc.head())

    print("\n=== Scenario Analysis: 20% attrition, 15% growth ===")
    sc = scenario_analysis(hc, "Engineering", attrition_rate=0.20, growth_target_pct=0.15)
    print(sc)

    total_needed = sc["new_hires_needed"].sum()
    print("\n=== Hiring Velocity ===")
    hv = hiring_velocity(total_needed, 12, offer_accept_rate=0.75)
    for k, v in hv.items():
        print(f"  {k}: {v}")

    print("\n=== Gap Analysis (6-month horizon) ===")
    gaps = gap_analysis(hc)
    print(gaps.to_string(index=False))
