# import necessary packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.proportion import confint_proportions_2indep
from scipy.optimize import brentq

# Read data

users = pd.read_csv("users.csv")
assignments = pd.read_csv("assignments.csv")
orders = pd.read_csv("orders.csv")

########### Quick checks on the data ###########

# Check for NULLs

print(f"NULLs in Users:\n{users.isna().sum()}\n")
print(f"NULLs in Assignments:\n{assignments.isna().sum()}\n")
print(f"NULLs in Orders:\n{orders.isna().sum()}\n")

# Dates

print(f"Max Order Date: {orders['ordered_at'].max()}")
print(f"Min Order Date: {orders['ordered_at'].min()}")
print(f"Max Assignment Date: {assignments['assigned_at'].max()}")
print(f"Min Assignment Date: {assignments['assigned_at'].min()}")   

# Check if all IDs are unique (one row = one user / one assignment / one order)

print(f"Uniqueness in Users table (user_id): {users['user_id'].is_unique}")
print(f"Uniqueness in Assignments table (user_id): {assignments['user_id'].is_unique}")   
print(f"Uniqueness in Orders table (order_id): {orders['order_id'].is_unique}")           

# Check if all users have an assignment

print(f"Do all users have an assignment?: {users['user_id'].isin(assignments['user_id']).all()}")   

# Check that every order belongs to a user who is in both users and assignments

orders_copy = orders.copy()
in_users = orders_copy["user_id"].isin(users["user_id"])
in_assignments = orders_copy["user_id"].isin(assignments["user_id"])
orders_copy["in_both"] = in_users & in_assignments

n_orphan_orders = (~orders_copy["in_both"]).sum()
print(f"Orders whose user is missing from users or assignments: {n_orphan_orders}")

########### Clean dataframe - check if users are assigned only variant or more ###########

# Ensure assigned_at is a datetime

assignments['assigned_at'] = pd.to_datetime(assignments['assigned_at'])

n_assignment_rows_raw = len(assignments)

# Count unique categories for each user

category_counts = assignments.groupby('user_id')['variant'].nunique()

# Users whose assignments all have the same category

same_category_users = category_counts[category_counts == 1].index

# Users with different categories

different_category_users = category_counts[category_counts > 1].index

# Same-variant duplicates = users with more than one row but a single variant

rows_per_user = assignments.groupby('user_id').size()
n_same_variant_dupes = ((rows_per_user > 1) & (category_counts == 1)).sum()

print(f"\nAssignment rows (raw): {n_assignment_rows_raw:,}")
print(f"Users logged twice in the same variant (keep earliest row): {n_same_variant_dupes}")
print(f"Users logged in both variants (excluded): {len(different_category_users)}")

# Remove all assignments for users with different categories

assignments = assignments[
    ~assignments['user_id'].isin(different_category_users)
].copy()

# Keep only the earliest assignment for each user

assignments = (
    assignments.sort_values('assigned_at')
    .drop_duplicates(subset='user_id', keep='first')
    .reset_index(drop=True)
)

print(f"Users in the analysis: {len(assignments):,}")

########### Sample Ratio Mismatch - Run a Chi-Square Goodness-of-Fit test ###########

def check_sample_ratio_mismatch(
    observed_counts: list[int],
    expected_ratios: list[float] = [0.5, 0.5],
    alpha: float = 0.01,
) -> dict:

    total_sample_size = sum(observed_counts)
    expected_counts = [ratio * total_sample_size for ratio in expected_ratios]

    chi2_stat, p_value = stats.chisquare(
        f_obs=observed_counts, f_exp=expected_counts
    )

    srm_detected = p_value < alpha

    return {
        "chi2_statistic": round(chi2_stat, 4),
        "p_value": p_value,
        "p_value_formatted": f"{p_value:.6f}",
        "srm_detected": srm_detected,
        "observed_counts": list(observed_counts),
        "expected_counts": [round(c, 2) for c in expected_counts],
        "conclusion": (
            "SRM DETECTED! Randomization is biased. Do not trust experiment results."
            if srm_detected
            else "NO SRM DETECTED. Traffic allocation is unbiased."
        ),
    }

arm_counts = assignments["variant"].value_counts()
observed_counts = [int(arm_counts["control"]), int(arm_counts["treatment"])]
result_srm = check_sample_ratio_mismatch(observed_counts)

print(f"\nObserved: {result_srm['observed_counts']}")
print(f"Expected: {result_srm['expected_counts']}")
print(f"p-value:  {result_srm['p_value_formatted']}")
print(f"Result:   {result_srm['conclusion']}\n")

########### Covariate Balance ###########

balance_df = assignments.merge(users, on="user_id", how="inner")

for col in ["device_type", "acquisition_channel", "customer_type"]:
    matrix = pd.crosstab(index=balance_df[col], columns=balance_df["variant"], normalize="columns")
    print(matrix.round(4))
    print()

########### Building the Analysis Table ###########

dataset_temp = pd.merge(assignments, users[['user_id','device_type', 'acquisition_channel', 'customer_type']], on='user_id', how='inner')

orders['ordered_at'] = pd.to_datetime(orders['ordered_at'])
assignments['assigned_at'] = pd.to_datetime(assignments['assigned_at'])

orders_merged = orders.merge(
    assignments[["user_id", "assigned_at"]], on="user_id", how="inner"
)

days_diff = (
    orders_merged["ordered_at"] - orders_merged["assigned_at"]
).dt.days

# keep orders from 0 up to (but not including) 7 full days after the user's own assignment

valid_orders = orders_merged[
    (days_diff >= 0) & (days_diff < 7)
].copy()                                   

valid_orders["margin"] = (
    np.where(valid_orders["returned"] == 1, 0.0, 0.38 * valid_orders["order_value"])
    + valid_orders["shipping_fee_charged"]
    - valid_orders["shipping_cost_to_company"]
)

order_metrics = (
    valid_orders.groupby("user_id")
    .agg(
        num_orders_7d=("order_id", "count"),
        revenue_7d=("order_value", "sum"),
        num_returned=("returned", "sum"),
        shipping_fee=("shipping_fee_charged", "sum"),
        shipping_cost=("shipping_cost_to_company", "sum"),
        margin_7d=("margin", "sum"),
        max_order_value=("order_value", "max"),
    )
    .reset_index()
)

final_df = dataset_temp.merge(order_metrics, on="user_id", how="left")

final_df["n_orders"] = final_df["num_orders_7d"].fillna(0).astype(int)
final_df["revenue"] = final_df["revenue_7d"].fillna(0.0)
final_df["n_returned"] = final_df["num_returned"].fillna(0).astype(int)
final_df["shipping_fee_total"] = final_df["shipping_fee"].fillna(0.0)
final_df["shipping_cost_total"] = final_df["shipping_cost"].fillna(0.0)
final_df["margin"] = final_df["margin_7d"].fillna(0.0)
final_df["max_order_value"] = final_df["max_order_value"].fillna(0.0)
final_df["converted"] = np.where(final_df["n_orders"] > 0, 1, 0)

final_df = final_df[[
    "user_id", "variant", "assigned_at", "device_type", "acquisition_channel",
    "customer_type", "n_orders", "revenue", "converted", "n_returned",
    "shipping_fee_total", "shipping_cost_total", "margin", "max_order_value",
]]

print(f"final_df rows: {len(final_df):,}  | unique users: {final_df['user_id'].nunique():,}")
print(f"Buyers: {final_df['converted'].sum():,}  | in-window orders: {final_df['n_orders'].sum():,}")


########### Two-Proportions z-test ###########

def run_two_proportion_ztest(
    successes_control: int,
    total_control: int,
    successes_treatment: int,
    total_treatment: int,
    alpha: float = 0.05,
) -> dict:

    counts = [successes_treatment, successes_control]
    nobs = [total_treatment, total_control]

    z_stat, p_value = proportions_ztest(
        count=counts, nobs=nobs, alternative="two-sided"
    )

    rate_control = successes_control / total_control
    rate_treatment = successes_treatment / total_treatment
    abs_lift_pp = (rate_treatment - rate_control) * 100          
    lift = (rate_treatment - rate_control) / rate_control

    is_significant = p_value < alpha

    return {
        "control_rate": round(rate_control * 100, 2),
        "treatment_rate": round(rate_treatment * 100, 2),
        "absolute_lift_pp": round(abs_lift_pp, 3),
        "relative_lift_pct": round(lift * 100, 2),
        "z_statistic": round(z_stat, 4),
        "p_value": p_value,
        "statistically_significant": is_significant,
        "conclusion": (
            "Statistically significant at the 5% level: reject the null of no effect. "
            "Judge the size of the effect from the confidence interval below."
            if is_significant
            else "Not statistically significant: fail to reject the null. "
                 "The difference could be random noise."
        ),
    }


control_users = len(final_df[final_df['variant'] == 'control'])
control_buyers = len(final_df.loc[lambda x: (x['n_orders'] > 0) & (x['variant'] == 'control')])

treatment_users = len(final_df[final_df['variant'] == 'treatment'])
treatment_buyers = len(final_df.loc[lambda x: (x['n_orders'] > 0) & (x['variant'] == 'treatment')])

result = run_two_proportion_ztest(
    control_buyers,
    control_users,
    treatment_buyers,
    treatment_users
)

print(f"\nControl Conversion Rate:   {result['control_rate']}%")
print(f"Treatment Conversion Rate: {result['treatment_rate']}%")
print(f"Absolute Lift:             {result['absolute_lift_pp']} pp")
print(f"Relative Lift:             {result['relative_lift_pct']}%")
print(f"Z-Statistic:               {result['z_statistic']}")
print(f"p-value:                   {result['p_value']:.4f}")
print(f"Result:                    {result['conclusion']}")


# Calculate Confidence Intervals

ci_lower, ci_upper = confint_proportions_2indep(
    count1=treatment_buyers,
    nobs1=treatment_users,
    count2=control_buyers,
    nobs2=control_users,
    method="wald",
    compare="diff",
    alpha=0.05
)

print(f"95% CI for the lift: {ci_lower * 100:.2f} pp to {ci_upper * 100:.2f} pp")


########### Secondary Metrics ###########

def welch_summary(df, col) -> dict:
    t = df[df["variant"] == "treatment"][col]
    c = df[df["variant"] == "control"][col]

    t_stat, p_value = stats.ttest_ind(t, c, equal_var=False)

    n_t, n_c = len(t), len(c)
    var_t, var_c = t.var(ddof=1), c.var(ddof=1)
    mean_diff = t.mean() - c.mean()
    se_diff = np.sqrt(var_t / n_t + var_c / n_c)

    welch_df = (var_t / n_t + var_c / n_c) ** 2 / (
        (var_t / n_t) ** 2 / (n_t - 1) + (var_c / n_c) ** 2 / (n_c - 1)
    )
    t_critical = stats.t.ppf(1 - 0.05 / 2, df=welch_df)

    return {
        "control_mean": c.mean(),
        "treatment_mean": t.mean(),
        "diff": mean_diff,
        "ci_lower": mean_diff - t_critical * se_diff,
        "ci_upper": mean_diff + t_critical * se_diff,
        "t_stat": t_stat,
        "p_value": p_value,                 
    }


def print_welch(label, res):
    print(f"{label}: control ${res['control_mean']:.3f} | treatment ${res['treatment_mean']:.3f} | "
          f"diff ${res['diff']:+.3f} | 95% CI [{res['ci_lower']:+.3f}, {res['ci_upper']:+.3f}] | "
          f"t = {res['t_stat']:.2f}, p = {res['p_value']:.4f}")


# Revenue / contribution margin per user, ALL users

print("\n--- Per-user metrics, all users ---")
res_rev_all = welch_summary(final_df, "revenue")
res_margin_all = welch_summary(final_df, "margin")
print_welch("Revenue per user", res_rev_all)
print_welch("Margin per user ", res_margin_all)


BULK_THRESHOLD = 500
bulk_users = final_df[final_df["max_order_value"] >= BULK_THRESHOLD]
print(f"\nBulk-order users (any order >= ${BULK_THRESHOLD}): {len(bulk_users)}  "
      f"{bulk_users['variant'].value_counts().to_dict()}")

df_no_bulk = final_df[final_df["max_order_value"] < BULK_THRESHOLD]

print("\n--- Per-user metrics, excluding bulk-order users ---")
res_rev_nb = welch_summary(df_no_bulk, "revenue")
res_margin_nb = welch_summary(df_no_bulk, "margin")
print_welch("Revenue per user", res_rev_nb)
print_welch("Margin per user ", res_margin_nb)


# Average Order Value (AOV): total revenue / total orders, by arm

def aov(df) -> dict:
    control = df[df['variant'] == 'control']
    treatment = df[df['variant'] == 'treatment']

    aov_control = control['revenue'].sum() / control['n_orders'].sum()
    aov_treatment = treatment['revenue'].sum() / treatment['n_orders'].sum()

    return {
        "aov_control": aov_control,
        "aov_treatment": aov_treatment
    }

aov_all = aov(final_df)
aov_nb = aov(df_no_bulk)

print("\n--- Average order value ---")
print(f"AOV, all users:            control ${aov_all['aov_control']:.2f} | treatment ${aov_all['aov_treatment']:.2f}")
print(f"AOV, excluding bulk users: control ${aov_nb['aov_control']:.2f} | treatment ${aov_nb['aov_treatment']:.2f}")


# Return Rate (returned orders / orders), excluding bulk users to match the AOV view

def return_rate(df) -> dict:
    control = df[df['variant'] == 'control']
    treatment = df[df['variant'] == 'treatment']

    returned_control = control['n_returned'].sum()
    orders_control = control['n_orders'].sum()
    returned_treatment = treatment['n_returned'].sum()
    orders_treatment = treatment['n_orders'].sum()

    return_rate_control = returned_control / orders_control
    return_rate_treatment = returned_treatment / orders_treatment

    z_stat, p_value = proportions_ztest(
        count=[returned_treatment, returned_control],
        nobs=[orders_treatment, orders_control],
        alternative="two-sided",
    )

    return {
        "Return Rate Control": return_rate_control,
        "Return Rate Treatment": return_rate_treatment,
        "z-stat": z_stat,
        "p-value": p_value
    }

return_rate_stat = return_rate(df_no_bulk)

print("\n--- Return rate ---")
print(f"Return Rate Control:   {return_rate_stat['Return Rate Control']:.2%}")
print(f"Return Rate Treatment: {return_rate_stat['Return Rate Treatment']:.2%}")
print(f"Z-Stat: {return_rate_stat['z-stat']:.3f}   p-value: {return_rate_stat['p-value']:.4f}")



N_SCALE = 1000000
print(f"\n--- Margin per user at {N_SCALE:,} users (all users) ---")
print(f"Point estimate: ${res_margin_all['diff'] * N_SCALE:+,.0f}   "
      f"95% range: ${res_margin_all['ci_lower'] * N_SCALE:+,.0f} to ${res_margin_all['ci_upper'] * N_SCALE:+,.0f}")


# Margin decomposition (merchandise margin vs. net shipping), per user

dec = final_df.copy()
dec["net_shipping"] = dec["shipping_fee_total"] - dec["shipping_cost_total"]
dec["merch_margin"] = dec["margin"] - dec["net_shipping"]          # margin = merch margin + net shipping

print("\n--- Margin decomposition per user (all users) ---")
print(dec.groupby("variant")[["merch_margin", "net_shipping", "margin"]].mean().round(3))
means = dec.groupby("variant")[["merch_margin", "net_shipping", "margin"]].mean()
print("Treatment minus control:")
print((means.loc["treatment"] - means.loc["control"]).round(3))


# Ship if conversion lift is significant at 5% AND margin per user is not significantly lower.

conv_ok = result["statistically_significant"] and result["absolute_lift_pp"] > 0
margin_sig_lower = (res_margin_all["p_value"] < 0.05) and (res_margin_all["diff"] < 0)
print("\n--- Decision rule ---")
print(f"Conversion lift significant and positive: {conv_ok}")
print(f"Margin per user significantly lower:      {margin_sig_lower}")
print(f"Rule says: {'SHIP' if conv_ok and not margin_sig_lower else 'DO NOT SHIP (or test more)'}")


########### Analyze Segments ###########

def lift_stats(df):
    c = df[df["variant"] == "control"]["converted"]
    t = df[df["variant"] == "treatment"]["converted"]
    n_c, n_t = len(c), len(t)
    x_c, x_t = c.sum(), t.sum()
    p_c, p_t = x_c / n_c, x_t / n_t

    diff = p_t - p_c
    se = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    p_pool = (x_t + x_c) / (n_t + n_c)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_t + 1 / n_c))
    z_stat = diff / se_pool
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return {
        "n_control": n_c, "n_treatment": n_t,
        "rate_control_%": p_c * 100, "rate_treatment_%": p_t * 100,
        "lift_pp": diff * 100,
        "ci_low_pp": (diff - 1.96 * se) * 100,
        "ci_high_pp": (diff + 1.96 * se) * 100,
        "p_value": p_value,
        "se_pp": se * 100,
    }

rows = [{"group": "Overall", "segment": "Overall", **lift_stats(final_df)}]
for col, label in [("device_type", "Device"), ("customer_type", "Customer")]:
    for value in sorted(final_df[col].unique()):
        sub = final_df[final_df[col] == value]
        rows.append({"group": label, "segment": f"{value}", **lift_stats(sub)})

seg = pd.DataFrame(rows)

n_tests = (seg["group"] != "Overall").sum()
seg["p_bonferroni"] = np.where(
    seg["group"] == "Overall", np.nan, np.minimum(seg["p_value"] * n_tests, 1.0)
)
print("\n" + seg.round(4).to_string(index=False))

for label in ["Device", "Customer"]:
    g = seg[seg["group"] == label]
    w = 1 / g["se_pp"] ** 2
    pooled_lift = (w * g["lift_pp"]).sum() / w.sum()
    q = (w * (g["lift_pp"] - pooled_lift) ** 2).sum()
    dof = len(g) - 1
    print(f"{label}: Q = {q:.3f}, df = {dof}, p = {stats.chi2.sf(q, dof):.3f}")

INK, INK_2, SERIES, GRID = "#0b0b0b", "#52514e", "#2a78d6", "#d9d8d4"

plot_df = seg.iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(8, 4.2))
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

for i, r in plot_df.iterrows():
    ax.hlines(i, r["ci_low_pp"], r["ci_high_pp"], color=SERIES, linewidth=2, zorder=2)
    ax.plot(r["lift_pp"], i, marker="D" if r["segment"] == "Overall" else "o",
            markersize=8, color=SERIES, markeredgecolor="#fcfcfb", markeredgewidth=2, zorder=3)
    ax.annotate(f"{r['lift_pp']:+.2f} pp", (r["ci_high_pp"], i), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=9, color=INK_2)

ax.axvline(0, color=INK_2, linewidth=1, zorder=1)
ax.set_yticks(range(len(plot_df)))
ax.set_yticklabels([(r["segment"] if r["group"] == "Overall" else f"{r['group']}: {r['segment']}")
                    + f"  (n={r['n_control'] + r['n_treatment']:,})"
                    for _, r in plot_df.iterrows()], color=INK)
ax.set_xlabel("Lift in conversion rate, treatment minus control (percentage points)", color=INK_2)
ax.set_title("Effect of the $50 free-shipping threshold, by segment", loc="left",
             color=INK, fontsize=12, fontweight="bold")
ax.grid(axis="x", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for side in ["top", "right", "left"]:
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="both", colors=INK_2, length=0)
ax.set_xlim(left=min(-0.4, plot_df["ci_low_pp"].min() - 0.15),
            right=plot_df["ci_high_pp"].max() + 0.6)
plt.tight_layout()
plt.savefig("forest_plot.png", dpi=150)
plt.show()

########### Power and Sample Size ###########

ALPHA, POWER = 0.05, 0.80
z_alpha = stats.norm.ppf(1 - ALPHA / 2)      
z_power = stats.norm.ppf(POWER)              

# Users needed per variant to detect a lift of `lift_pp` from a baseline rate

def sample_size_per_arm(baseline, lift_pp):
    p1 = baseline
    p2 = baseline + lift_pp / 100
    variance = p1 * (1 - p1) + p2 * (1 - p2)
    return (z_alpha + z_power) ** 2 * variance / (p2 - p1) ** 2

# Power for a given lift and a given number of users per variant

def power_for(baseline, lift_pp, n_per_arm):
    p1 = baseline
    p2 = baseline + lift_pp / 100
    se = np.sqrt(p1 * (1 - p1) / n_per_arm + p2 * (1 - p2) / n_per_arm)
    return stats.norm.cdf(abs(p2 - p1) / se - z_alpha)

# Minimum detectable effect (MDE): the smallest lift this sample could detect with 80% power

def mde_pp(baseline, n_per_arm):
    return brentq(lambda d: power_for(baseline, d, n_per_arm) - POWER, 0.001, 10)

# What could this experiment detect?

control = final_df[final_df["variant"] == "control"]
n_arm = final_df.groupby("variant").size().min()
base_overall = control["converted"].mean()

print(f"\nOverall: baseline {base_overall:.2%}, ~{n_arm:,} users per arm")
print(f"  MDE at 80% power: {mde_pp(base_overall, n_arm):.2f} pp   (observed lift was {result['absolute_lift_pp']:+.2f} pp)")

print("\nMDE by segment (80% power):")
seg_rows = []
for col in ["device_type", "customer_type"]:
    for value in sorted(final_df[col].unique()):
        sub = final_df[final_df[col] == value]
        n_seg = sub.groupby("variant").size().min()
        base = sub[sub["variant"] == "control"]["converted"].mean()
        seg_rows.append({"segment": f"{col}: {value}", "users_per_arm": n_seg,
                         "baseline_%": base * 100, "MDE_pp": mde_pp(base, n_seg)})
print(pd.DataFrame(seg_rows).round(2).to_string(index=False))

# How big should a mobile-only follow-up test be?

mobile = final_df[final_df["device_type"] == "mobile"]
base_mobile = mobile[mobile["variant"] == "control"]["converted"].mean()

print(f"\nMobile baseline conversion: {base_mobile:.2%}")
print("Users needed per arm to detect each lift (80% power, 5% level):")
lifts = [0.25, 0.50, 0.75, 1.00]
needed = {l: sample_size_per_arm(base_mobile, l) for l in lifts}
for l in lifts:
    print(f"  +{l:.2f} pp -> {np.ceil(needed[l]):,.0f} per arm  ({2 * np.ceil(needed[l]):,.0f} total)")

# How long would that take? Use the experiment's own daily mobile traffic

days = (final_df["assigned_at"].max().normalize() - final_df["assigned_at"].min().normalize()).days + 1
mobile_per_day = len(mobile) / days
print(f"\nMobile users entering the test per day (this experiment): {mobile_per_day:,.0f}")
for l in lifts:
    total = 2 * np.ceil(needed[l])
    print(f"  +{l:.2f} pp -> about {total / mobile_per_day:.0f} days")

# Required sample size vs. the lift you want to detect

grid_lifts = np.linspace(0.2, 1.2, 60)
fig, ax = plt.subplots(figsize=(7, 4))
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")
ax.plot(grid_lifts, [sample_size_per_arm(base_mobile, l) for l in grid_lifts],
        color=SERIES, linewidth=2)
ax.axvline(0.5, color=INK_2, linewidth=1, linestyle=":")
ax.annotate(f"+0.5 pp needs ~{sample_size_per_arm(base_mobile, 0.5):,.0f} per arm",
            (0.5, sample_size_per_arm(base_mobile, 0.5)), xytext=(10, 10),
            textcoords="offset points", fontsize=9, color=INK_2)
ax.set_xlabel("Lift you want to detect (percentage points)", color=INK_2)
ax.set_ylabel("Users needed per arm", color=INK_2)
ax.set_title("Smaller effects need far more users (mobile baseline)", loc="left",
             color=INK, fontsize=12, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.grid(axis="y", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)
for side in ["left", "bottom"]:
    ax.spines[side].set_color(GRID)
ax.tick_params(colors=INK_2, length=0)
plt.tight_layout()
plt.savefig("power_curve.png", dpi=150)
plt.show()