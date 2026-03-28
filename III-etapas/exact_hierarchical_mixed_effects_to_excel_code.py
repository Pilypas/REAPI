# exact_hierarchical_mixed_effects_fallback.py
#
# PURPOSE
# -------
# This script fits the exact hierarchical mixed-effects fallback model in Python
# and exports all main fitted components into Excel.
#
# MODEL
# -----
# Let:
#   c = country
#   r = region
#   t = time
#   y = variable to be imputed
#
# The cubic hierarchical fallback model is:
#
#   y_rct = beta0 + beta1*t_c + beta2*t_c^2 + beta3*t_c^3 + v_c + u_r(c) + e_rct
#
# where:
#   t_c     = centered time
#   v_c     = country-level random intercept
#   u_r(c)  = region-level random intercept nested within country
#
# This script:
#   1) reads a long-format panel from Excel,
#   2) optionally performs internal linear interpolation within each region,
#   3) classifies regions by data support,
#   4) fits the exact mixed-effects model by REML,
#   5) extracts fixed effects, country random intercepts, and region random intercepts,
#   6) predicts values for all rows,
#   7) fills missing values only in sparse or completely empty regions,
#   8) exports all steps and outputs into Excel.
#
# REQUIRED INPUT COLUMNS
# ----------------------
#   c : country identifier
#   r : region identifier
#   t : time
#   y : variable to impute
#
# REQUIRED PACKAGES
# -----------------
#   pip install pandas numpy statsmodels openpyxl
#
# IMPORTANT NOTES
# ---------------
# 1) This script estimates the exact hierarchical mixed-effects model in Python.
#    The Excel file is only the exported result, not the estimation engine.
#
# 2) By default, the model is fitted on:
#       observed y values
#       + internal linear interpolations inside each region
#    This follows the idea that internal gaps can first be filled safely,
#    while leading and trailing gaps remain for the mixed-effects fallback.
#
# 3) If you want the model to be fitted only on truly observed values,
#    set:
#       USE_STAGE1_INTERPOLATION_FOR_FIT = False
#
# 4) The final fallback-imputed series keeps:
#       original observed values,
#       internal interpolations,
#       and uses mixed-model predictions only for the remaining missing values
#       in sparse or completely empty regions.
#
# 5) For a completely unobserved region, there is no region-specific effect
#    that can be learned from its own data. In that case, prediction uses:
#       fixed cubic trend + country random intercept
#    If the country itself has no usable fitted effect, prediction falls back to:
#       fixed cubic trend only

import re
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# =========================================================
# USER SETTINGS
# =========================================================

INPUT_EXCEL_PATH = "Book1.xlsx"
INPUT_SHEET_NAME = 0              # sheet name or sheet index
OUTPUT_EXCEL_PATH = "exact_hierarchical_mixed_effects_output.xlsx"

COUNTRY_COL = "c"
REGION_COL = "r"
TIME_COL = "t"
VALUE_COL = "y"

# Minimum number of observed values required for a region to be treated
# as sufficiently supported.
# Example:
#   minobs = 9
# means:
#   0 observations     -> all-missing region
#   1 to 8 observations -> sparse region
#   9 or more          -> sufficiently observed region
MINOBS = 9

# If True, the model fitting sample uses:
#   observed y values + internal linear interpolations within each region
# If False, the model fitting sample uses only observed y values.
USE_STAGE1_INTERPOLATION_FOR_FIT = True

# If True, final fallback predictions are used only for rows that belong to:
#   sparse regions or all-missing regions
# If False, predictions are calculated for all rows, but original y and stage-1
# values are still preserved.
APPLY_FALLBACK_ONLY_TO_SPARSE_OR_EMPTY = True


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def internal_linear_interpolation(series: pd.Series) -> pd.Series:
    """
    Fill only internal gaps.
    Missing values at the beginning or end are not extrapolated.
    """
    return series.interpolate(method="linear", limit_area="inside")


def make_nested_region_id(country_value, region_value) -> str:
    """
    Create a unique nested region identifier.
    This ensures that region codes are unique within country.
    """
    return f"{country_value}__{region_value}"


def choose_base_series_for_fit(df: pd.DataFrame) -> str:
    """
    Return the column name that should be used for model fitting.
    """
    if USE_STAGE1_INTERPOLATION_FOR_FIT:
        return "y_stage1"
    return VALUE_COL


def safe_string(x) -> str:
    """
    Convert values to string safely.
    """
    if pd.isna(x):
        return ""
    return str(x)


def extract_country_and_region_effects(mixed_result, fit_df, country_col, nested_region_col):
    """
    Extract country-level and region-level random intercepts from statsmodels output.

    The MixedLM specification used below is:
        - country random intercept through groups=country
        - region nested random intercept through vc_formula

    statsmodels stores these in mixed_result.random_effects.

    This function returns:
        country_effects_df
        region_effects_df
    """

    random_effects = mixed_result.random_effects

    observed_countries = pd.Series(fit_df[country_col].unique()).dropna().tolist()
    observed_nested_regions = pd.Series(fit_df[nested_region_col].unique()).dropna().tolist()

    country_rows = []
    region_rows = []

    for country in observed_countries:
        if country not in random_effects:
            continue

        series = random_effects[country]

        # The first element is the country-level random intercept.
        # In statsmodels MixedLM with re_formula="1", the first element is
        # the group-specific random intercept.
        if isinstance(series, pd.Series) and len(series) > 0:
            country_effect = float(series.iloc[0])
        else:
            country_effect = 0.0

        country_rows.append({
            country_col: country,
            "country_random_intercept": country_effect
        })

        # Additional entries correspond to variance-component effects.
        # We search for nested region IDs in the index labels.
        if isinstance(series, pd.Series):
            for idx_label, value in series.items():
                idx_text = safe_string(idx_label)

                # Skip the first country-level intercept if index label corresponds
                # to the primary random effect.
                if idx_text == safe_string(series.index[0]):
                    continue

                # Try to find which nested region ID appears in this label.
                # Example possible labels:
                #   region[C(region_nested)[LT01__R1]]
                # We search for any known nested ID inside the label text.
                matched_nested_region = None
                for nested_region in observed_nested_regions:
                    if nested_region in idx_text:
                        matched_nested_region = nested_region
                        break

                if matched_nested_region is not None:
                    region_rows.append({
                        country_col: country,
                        nested_region_col: matched_nested_region,
                        "region_random_intercept": float(value),
                        "raw_random_effect_label": idx_text
                    })

    country_effects_df = pd.DataFrame(country_rows)
    region_effects_df = pd.DataFrame(region_rows)

    # Remove duplicates if any appear in parsing
    if not country_effects_df.empty:
        country_effects_df = country_effects_df.drop_duplicates(subset=[country_col])

    if not region_effects_df.empty:
        region_effects_df = region_effects_df.drop_duplicates(subset=[country_col, nested_region_col])

    return country_effects_df, region_effects_df


def build_model_formula():
    """
    Build the fixed-effect formula for the cubic global trend.
    """
    return "y_fit ~ t_centered + t_centered2 + t_centered3"


def format_model_summary_text(mixed_result) -> pd.DataFrame:
    """
    Export the model summary as plain text lines into a dataframe for Excel.
    """
    summary_text = str(mixed_result.summary())
    lines = summary_text.splitlines()
    return pd.DataFrame({"model_summary": lines})


def add_basic_check(df: pd.DataFrame, message: str) -> pd.DataFrame:
    return pd.DataFrame({"check": [message]})


# =========================================================
# MAIN PROCEDURE
# =========================================================

def run_exact_hierarchical_fallback():
    # -----------------------------------------------------
    # 1) READ DATA
    # -----------------------------------------------------
    df = pd.read_excel(INPUT_EXCEL_PATH, sheet_name=INPUT_SHEET_NAME)

    required_cols = [COUNTRY_COL, REGION_COL, TIME_COL, VALUE_COL]
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(
            f"Missing required columns: {missing_required}. "
            f"Your input file must contain {required_cols}."
        )

    df = df.copy()
    df = df.sort_values([COUNTRY_COL, REGION_COL, TIME_COL]).reset_index(drop=True)

    # -----------------------------------------------------
    # 2) CREATE USEFUL IDENTIFIERS
    # -----------------------------------------------------
    df["country_str"] = df[COUNTRY_COL].astype(str)
    df["region_str"] = df[REGION_COL].astype(str)
    df["region_nested"] = df.apply(
        lambda row: make_nested_region_id(row[COUNTRY_COL], row[REGION_COL]),
        axis=1
    )

    # -----------------------------------------------------
    # 3) ORIGINAL MISSINGNESS
    # -----------------------------------------------------
    df["y_missing_original"] = df[VALUE_COL].isna()

    # -----------------------------------------------------
    # 4) COUNT OBSERVATIONS BY REGION
    # -----------------------------------------------------
    df["nonmissing_by_region"] = (
        df.groupby([COUNTRY_COL, REGION_COL])[VALUE_COL]
          .transform(lambda s: s.notna().sum())
    )

    df["all_missing_region"] = df["nonmissing_by_region"] == 0
    df["low_obs_region"] = (
        (df["nonmissing_by_region"] > 0) &
        (df["nonmissing_by_region"] < MINOBS)
    )
    df["sufficient_obs_region"] = df["nonmissing_by_region"] >= MINOBS

    # -----------------------------------------------------
    # 5) STAGE 1: INTERNAL LINEAR INTERPOLATION
    # -----------------------------------------------------
    df["y_stage1"] = (
        df.groupby([COUNTRY_COL, REGION_COL])[VALUE_COL]
          .transform(internal_linear_interpolation)
    )

    df["filled_by_stage1"] = (
        df[VALUE_COL].isna() &
        df["y_stage1"].notna()
    )

    # -----------------------------------------------------
    # 6) PREPARE MODEL-FITTING SERIES
    # -----------------------------------------------------
    base_fit_col = choose_base_series_for_fit(df)
    df["y_fit"] = df[base_fit_col]

    # -----------------------------------------------------
    # 7) TIME TRANSFORMATION FOR CUBIC GLOBAL TREND
    # -----------------------------------------------------
    df["t_numeric"] = pd.to_numeric(df[TIME_COL], errors="coerce")
    if df["t_numeric"].isna().any():
        raise ValueError(
            f"Column '{TIME_COL}' must be numeric or convertible to numeric."
        )

    t_mean = df["t_numeric"].mean()
    df["t_centered"] = df["t_numeric"] - t_mean
    df["t_centered2"] = df["t_centered"] ** 2
    df["t_centered3"] = df["t_centered"] ** 3

    # -----------------------------------------------------
    # 8) DEFINE FITTING SAMPLE
    # -----------------------------------------------------
    fit_df = df.loc[df["y_fit"].notna(), [
        COUNTRY_COL, REGION_COL, "region_nested",
        TIME_COL, "t_numeric", "t_centered", "t_centered2", "t_centered3",
        VALUE_COL, "y_stage1", "y_fit"
    ]].copy()

    if fit_df.empty:
        raise ValueError("No usable observations available to fit the model.")

    # -----------------------------------------------------
    # 9) FIT EXACT HIERARCHICAL MIXED-EFFECTS MODEL
    # -----------------------------------------------------
    # Fixed part:
    #   cubic global trend
    #
    # Random part:
    #   country random intercept -> groups=country
    #   region nested random intercept -> vc_formula with nested region id
    #
    formula = build_model_formula()

    mixed_model = smf.mixedlm(
        formula=formula,
        data=fit_df,
        groups=fit_df[COUNTRY_COL],
        re_formula="1",
        vc_formula={"region": "0 + C(region_nested)"}
    )

    # Try a robust sequence of optimizers
    try:
        mixed_result = mixed_model.fit(
            reml=True,
            method="lbfgs",
            maxiter=2000,
            disp=False
        )
    except Exception:
        mixed_result = mixed_model.fit(
            reml=True,
            method="powell",
            maxiter=2000,
            disp=False
        )

    # -----------------------------------------------------
    # 10) FIXED-EFFECT COMPONENTS
    # -----------------------------------------------------
    fe = mixed_result.fe_params

    beta0 = float(fe.get("Intercept", 0.0))
    beta1 = float(fe.get("t_centered", 0.0))
    beta2 = float(fe.get("t_centered2", 0.0))
    beta3 = float(fe.get("t_centered3", 0.0))

    df["fixed_global_trend"] = (
        beta0
        + beta1 * df["t_centered"]
        + beta2 * df["t_centered2"]
        + beta3 * df["t_centered3"]
    )

    fixed_effects_df = pd.DataFrame({
        "term": ["Intercept", "t_centered", "t_centered2", "t_centered3"],
        "estimate": [beta0, beta1, beta2, beta3]
    })

    # -----------------------------------------------------
    # 11) VARIANCE COMPONENTS
    # -----------------------------------------------------
    # Country-level random intercept variance
    country_variance = float(mixed_result.cov_re.iloc[0, 0]) if mixed_result.cov_re.shape == (1, 1) else np.nan

    # Region nested random intercept variance components
    # mixed_result.vcomp is aligned with vc_formula order
    region_variance = np.nan
    try:
        if hasattr(mixed_result, "vcomp") and len(mixed_result.vcomp) > 0:
            region_variance = float(mixed_result.vcomp[0])
    except Exception:
        pass

    residual_variance = float(mixed_result.scale)

    variance_components_df = pd.DataFrame({
        "component": [
            "country_random_intercept_variance",
            "region_nested_random_intercept_variance",
            "residual_variance"
        ],
        "estimate": [
            country_variance,
            region_variance,
            residual_variance
        ]
    })

    # -----------------------------------------------------
    # 12) EXTRACT COUNTRY AND REGION RANDOM INTERCEPTS
    # -----------------------------------------------------
    country_effects_df, region_effects_df = extract_country_and_region_effects(
        mixed_result=mixed_result,
        fit_df=fit_df,
        country_col=COUNTRY_COL,
        nested_region_col="region_nested"
    )

    if country_effects_df.empty:
        country_effects_df = pd.DataFrame(columns=[COUNTRY_COL, "country_random_intercept"])

    if region_effects_df.empty:
        region_effects_df = pd.DataFrame(columns=[COUNTRY_COL, "region_nested", "region_random_intercept", "raw_random_effect_label"])

    # Add plain region code back
    if not region_effects_df.empty:
        region_effects_df["region_code_from_nested"] = region_effects_df["region_nested"].apply(
            lambda x: str(x).split("__", 1)[1] if "__" in str(x) else str(x)
        )

    # -----------------------------------------------------
    # 13) MERGE RANDOM EFFECTS INTO FULL PANEL
    # -----------------------------------------------------
    df = df.merge(country_effects_df, on=COUNTRY_COL, how="left")
    df = df.merge(
        region_effects_df[["region_nested", "region_random_intercept"]],
        on="region_nested",
        how="left"
    )

    df["country_random_intercept"] = df["country_random_intercept"].fillna(0.0)
    df["region_random_intercept"] = df["region_random_intercept"].fillna(0.0)

    # -----------------------------------------------------
    # 14) BUILD FULL PREDICTION
    # -----------------------------------------------------
    df["exact_mixed_prediction"] = (
        df["fixed_global_trend"]
        + df["country_random_intercept"]
        + df["region_random_intercept"]
    )

    # -----------------------------------------------------
    # 15) FINAL FALLBACK SERIES
    # -----------------------------------------------------
    # Start from stage 1:
    #   observed values are preserved
    #   internal interpolations are preserved
    #
    df["y_after_exact_fallback"] = df["y_stage1"]

    if APPLY_FALLBACK_ONLY_TO_SPARSE_OR_EMPTY:
        fallback_fill_condition = (
            df["y_after_exact_fallback"].isna() &
            (df["low_obs_region"] | df["all_missing_region"])
        )
    else:
        fallback_fill_condition = df["y_after_exact_fallback"].isna()

    df.loc[fallback_fill_condition, "y_after_exact_fallback"] = df.loc[
        fallback_fill_condition, "exact_mixed_prediction"
    ]

    df["filled_by_exact_mixed_fallback"] = (
        fallback_fill_condition &
        df["y_after_exact_fallback"].notna()
    )

    # -----------------------------------------------------
    # 16) BUILD SHEETS FOR EXCEL EXPORT
    # -----------------------------------------------------
    methodology_df = pd.DataFrame({
        "item": [
            "Model type",
            "Global trend",
            "Country effect",
            "Region effect",
            "Fitting sample",
            "Final fallback fill rule"
        ],
        "description": [
            "Exact hierarchical mixed-effects model estimated in Python by REML",
            "3rd order polynomial (cubic) in centered time",
            "Country-level random intercept",
            "Region-level random intercept nested within country",
            "Observed y values, or observed + internal interpolation if selected",
            "Fill remaining missing values only in sparse or all-missing regions"
        ]
    })

    parameters_df = pd.DataFrame({
        "parameter": [
            "INPUT_EXCEL_PATH",
            "INPUT_SHEET_NAME",
            "OUTPUT_EXCEL_PATH",
            "COUNTRY_COL",
            "REGION_COL",
            "TIME_COL",
            "VALUE_COL",
            "MINOBS",
            "USE_STAGE1_INTERPOLATION_FOR_FIT",
            "APPLY_FALLBACK_ONLY_TO_SPARSE_OR_EMPTY"
        ],
        "value": [
            INPUT_EXCEL_PATH,
            str(INPUT_SHEET_NAME),
            OUTPUT_EXCEL_PATH,
            COUNTRY_COL,
            REGION_COL,
            TIME_COL,
            VALUE_COL,
            MINOBS,
            USE_STAGE1_INTERPOLATION_FOR_FIT,
            APPLY_FALLBACK_ONLY_TO_SPARSE_OR_EMPTY
        ]
    })

    # Main panel output with the most important columns
    panel_output_cols = [
        COUNTRY_COL, REGION_COL, TIME_COL, VALUE_COL,
        "y_missing_original",
        "nonmissing_by_region",
        "all_missing_region",
        "low_obs_region",
        "sufficient_obs_region",
        "y_stage1",
        "filled_by_stage1",
        "t_centered",
        "t_centered2",
        "t_centered3",
        "fixed_global_trend",
        "country_random_intercept",
        "region_random_intercept",
        "exact_mixed_prediction",
        "y_after_exact_fallback",
        "filled_by_exact_mixed_fallback"
    ]
    panel_output_df = df[panel_output_cols].copy()

    missing_only_df = df.loc[df["y_missing_original"], panel_output_cols].copy()

    checks_df = pd.concat([
        add_basic_check(df, f"Total rows: {len(df)}"),
        add_basic_check(df, f"Rows used in model fitting: {len(fit_df)}"),
        add_basic_check(df, f"Originally missing rows: {int(df['y_missing_original'].sum())}"),
        add_basic_check(df, f"Filled by stage 1 interpolation: {int(df['filled_by_stage1'].sum())}"),
        add_basic_check(df, f"Filled by exact mixed fallback: {int(df['filled_by_exact_mixed_fallback'].sum())}")
    ], ignore_index=True)

    model_summary_df = format_model_summary_text(mixed_result)

    # -----------------------------------------------------
    # 17) WRITE ALL OUTPUTS TO EXCEL
    # -----------------------------------------------------
    with pd.ExcelWriter(OUTPUT_EXCEL_PATH, engine="openpyxl") as writer:
        methodology_df.to_excel(writer, sheet_name="Methodology", index=False)
        parameters_df.to_excel(writer, sheet_name="Parameters", index=False)
        df[[COUNTRY_COL, REGION_COL, TIME_COL, VALUE_COL]].to_excel(writer, sheet_name="Input_Data", index=False)
        fit_df.to_excel(writer, sheet_name="Fitting_Sample", index=False)
        fixed_effects_df.to_excel(writer, sheet_name="Fixed_Effects", index=False)
        variance_components_df.to_excel(writer, sheet_name="Variance_Components", index=False)
        country_effects_df.to_excel(writer, sheet_name="Country_Random_Effects", index=False)
        region_effects_df.to_excel(writer, sheet_name="Region_Random_Effects", index=False)
        panel_output_df.to_excel(writer, sheet_name="Panel_Output", index=False)
        missing_only_df.to_excel(writer, sheet_name="Missing_Only", index=False)
        checks_df.to_excel(writer, sheet_name="Checks", index=False)
        model_summary_df.to_excel(writer, sheet_name="Model_Summary", index=False)

    print("Finished successfully.")
    print(f"Output written to: {OUTPUT_EXCEL_PATH}")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    run_exact_hierarchical_fallback()