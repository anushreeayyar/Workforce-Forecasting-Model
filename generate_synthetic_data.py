"""
generate_synthetic_data.py
--------------------------
Generates three synthetic workforce datasets mimicking real HR data:
  1. headcount_snapshots.csv  — monthly headcount by department
  2. employee_data.csv        — individual employee records (anonymized)
  3. ta_funnel.csv            — monthly recruiting pipeline by department

Run: python generate_synthetic_data.py
Outputs land in the /data folder.
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
import random
import os

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ── Config ───────────────────────────────────────────────────────────────────
START_DATE   = date(2022, 1, 1)
MONTHS       = 36          # 3 years of history
OUTPUT_DIR   = "data"

DEPARTMENTS = {
    "Engineering":       {"base_hc": 120, "growth_rate": 0.018, "attrition_rate": 0.013, "avg_salary": 145_000},
    "Sales":             {"base_hc":  80, "growth_rate": 0.015, "attrition_rate": 0.017, "avg_salary": 110_000},
    "Product":           {"base_hc":  40, "growth_rate": 0.012, "attrition_rate": 0.010, "avg_salary": 135_000},
    "HR":                {"base_hc":  25, "growth_rate": 0.008, "attrition_rate": 0.009, "avg_salary":  95_000},
    "Finance":           {"base_hc":  30, "growth_rate": 0.007, "attrition_rate": 0.008, "avg_salary": 105_000},
    "Operations":        {"base_hc":  60, "growth_rate": 0.010, "attrition_rate": 0.014, "avg_salary":  85_000},
    "Customer Success":  {"base_hc":  50, "growth_rate": 0.014, "attrition_rate": 0.016, "avg_salary":  80_000},
    "Data":              {"base_hc":  35, "growth_rate": 0.020, "attrition_rate": 0.011, "avg_salary": 140_000},
}

LEVELS = ["IC1", "IC2", "IC3", "IC4", "IC5", "Manager", "Director"]
LEVEL_SALARY_MULT = {"IC1": 0.65, "IC2": 0.80, "IC3": 1.00, "IC4": 1.20, "IC5": 1.45,
                     "Manager": 1.40, "Director": 1.90}
LEVEL_DIST = [0.15, 0.22, 0.28, 0.18, 0.08, 0.07, 0.02]   # probability weights

EXIT_REASONS = ["Voluntary", "Voluntary", "Voluntary", "Involuntary", "Retirement"]

# ── Helpers ───────────────────────────────────────────────────────────────────
def add_months(d: date, months: int) -> date:
    month = d.month - 1 + months
    year  = d.year + month // 12
    month = month % 12 + 1
    return d.replace(year=year, month=month)

def seasonal_factor(month_num: int) -> float:
    """Q1 and Q3 are peak hiring seasons."""
    seasonal = {1: 1.20, 2: 1.10, 3: 1.05, 4: 0.95,
                5: 0.90, 6: 0.85, 7: 1.10, 8: 1.05,
                9: 1.00, 10: 0.90, 11: 0.85, 12: 0.70}
    return seasonal.get(month_num, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. HEADCOUNT SNAPSHOTS
# ═══════════════════════════════════════════════════════════════════════════════
def generate_headcount_snapshots() -> pd.DataFrame:
    rows = []
    for dept, cfg in DEPARTMENTS.items():
        hc = cfg["base_hc"]
        for m in range(MONTHS):
            period      = add_months(START_DATE, m)
            g           = cfg["growth_rate"]
            a           = cfg["attrition_rate"]
            sf          = seasonal_factor(period.month)

            target_hc   = int(cfg["base_hc"] * (1 + g) ** m)
            gap         = max(0, target_hc - hc)
            new_hires   = int((gap + hc * g) * sf * np.random.uniform(0.85, 1.15))
            attritions  = max(0, int(hc * a * np.random.uniform(0.70, 1.30)))
            hc          = max(1, hc + new_hires - attritions)
            open_reqs   = max(0, int(gap * np.random.uniform(0.8, 1.2)))

            rows.append({
                "month":            period.strftime("%Y-%m"),
                "department":       dept,
                "headcount_actual": hc,
                "headcount_target": target_hc,
                "new_hires":        new_hires,
                "attritions":       attritions,
                "open_reqs":        open_reqs,
                "avg_salary":       int(cfg["avg_salary"] * np.random.uniform(0.95, 1.05)),
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. EMPLOYEE DATA
# ═══════════════════════════════════════════════════════════════════════════════
def generate_employee_data() -> pd.DataFrame:
    """Generates synthetic individual employee records."""
    rows  = []
    emp_id = 1000

    for dept, cfg in DEPARTMENTS.items():
        # Active employees at snapshot end
        n_active = int(cfg["base_hc"] * (1 + cfg["growth_rate"]) ** (MONTHS - 1))
        for _ in range(n_active):
            emp_id += 1
            level  = np.random.choice(LEVELS, p=LEVEL_DIST)
            salary = int(cfg["avg_salary"] * LEVEL_SALARY_MULT[level]
                         * np.random.uniform(0.90, 1.10))
            # Random hire date in the past 5 years
            days_back   = np.random.randint(30, 5 * 365)
            hire_date   = date.today() - timedelta(days=int(days_back))
            rows.append({
                "employee_id":  emp_id,
                "department":   dept,
                "level":        level,
                "hire_date":    hire_date.strftime("%Y-%m-%d"),
                "exit_date":    None,
                "exit_reason":  None,
                "annual_salary": salary,
                "status":       "Active",
            })

        # Former employees who exited during history window
        n_exited = int(n_active * cfg["attrition_rate"] * MONTHS)
        for _ in range(n_exited):
            emp_id   += 1
            level     = np.random.choice(LEVELS, p=LEVEL_DIST)
            salary    = int(cfg["avg_salary"] * LEVEL_SALARY_MULT[level]
                            * np.random.uniform(0.90, 1.10))
            month_idx = np.random.randint(0, MONTHS)
            exit_dt   = add_months(START_DATE, month_idx)
            hire_back = np.random.randint(60, 365 * 3)
            hire_dt   = exit_dt - timedelta(days=int(hire_back))
            rows.append({
                "employee_id":  emp_id,
                "department":   dept,
                "level":        level,
                "hire_date":    hire_dt.strftime("%Y-%m-%d"),
                "exit_date":    exit_dt.strftime("%Y-%m-%d"),
                "exit_reason":  random.choice(EXIT_REASONS),
                "annual_salary": salary,
                "status":       "Exited",
            })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TA FUNNEL
# ═══════════════════════════════════════════════════════════════════════════════
def generate_ta_funnel(hc_df: pd.DataFrame) -> pd.DataFrame:
    """Recruiting pipeline tied to headcount open_reqs."""
    rows = []
    for _, row in hc_df.iterrows():
        dept_cfg     = DEPARTMENTS[row["department"]]
        open_reqs    = max(row["open_reqs"], 1)
        applications = int(open_reqs * np.random.uniform(8, 15))
        interview_r  = np.random.uniform(0.25, 0.40)
        interviews   = int(applications * interview_r)
        offer_r      = np.random.uniform(0.20, 0.35)
        offers_made  = int(interviews * offer_r)
        accept_r     = np.random.uniform(0.65, 0.85)
        offers_acc   = int(offers_made * accept_r)
        # time-to-fill increases when pipeline is thin
        ttf_base     = 35 if dept_cfg["avg_salary"] > 120_000 else 25
        ttf          = int(ttf_base + max(0, 5 - offers_made) * 3
                           + np.random.normal(0, 5))
        rows.append({
            "month":             row["month"],
            "department":        row["department"],
            "roles_open":        open_reqs,
            "applications":      applications,
            "interviews":        interviews,
            "offers_made":       offers_made,
            "offers_accepted":   offers_acc,
            "offer_accept_rate": round(offers_acc / max(offers_made, 1), 3),
            "time_to_fill_days": max(10, ttf),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("📊 Generating headcount snapshots...")
    hc_df = generate_headcount_snapshots()
    hc_df.to_csv(f"{OUTPUT_DIR}/headcount_snapshots.csv", index=False)
    print(f"   ✓ {len(hc_df):,} rows → {OUTPUT_DIR}/headcount_snapshots.csv")

    print("👤 Generating employee data...")
    emp_df = generate_employee_data()
    emp_df.to_csv(f"{OUTPUT_DIR}/employee_data.csv", index=False)
    print(f"   ✓ {len(emp_df):,} rows → {OUTPUT_DIR}/employee_data.csv")

    print("🎯 Generating TA funnel data...")
    ta_df = generate_ta_funnel(hc_df)
    ta_df.to_csv(f"{OUTPUT_DIR}/ta_funnel.csv", index=False)
    print(f"   ✓ {len(ta_df):,} rows → {OUTPUT_DIR}/ta_funnel.csv")

    print("\n✅ All datasets generated successfully!")
    print(f"\nDataset summary:")
    print(f"  Departments: {len(DEPARTMENTS)}")
    print(f"  History window: {START_DATE} → {add_months(START_DATE, MONTHS-1)}")
    print(f"  Total headcount records: {len(hc_df):,}")
    print(f"  Total employee records:  {len(emp_df):,}")
    print(f"  Total TA funnel records: {len(ta_df):,}")
