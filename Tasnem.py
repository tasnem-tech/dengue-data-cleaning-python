import pandas as pd

# =========================
# Activity 1: Load + Inspect
# =========================

# 1) Load the dataset (EDIT the path if needed)
df = pd.read_csv(r"C:\Users\User\Downloads\Tasnem Week4_Data.csv")

# Extra: keep a raw backup so you never overwrite it
raw = df.copy()

print("Activity 1: Basic inspection")
print("Head:")
print(df.head(), "\n")

print("Shape (rows, cols):", df.shape, "\n")

print("Info:")
df.info()
print("\n")


# Helper: find a column name ignoring case/spaces/underscores
def find_col(possible_names):
    norm_map = {c: c.lower().replace(" ", "").replace("_", "") for c in df.columns}
    target = {n.lower().replace(" ", "").replace("_", "") for n in possible_names}
    for original, normed in norm_map.items():
        if normed in target:
            return original
    return None


# =========================
# Activity 2: Columns + Types
# =========================
print("Activity 2: Columns and types")
print("Columns:")
print(list(df.columns), "\n")

print("dtypes:")
print(df.dtypes, "\n")


# ==========================================
# Activity 3: Identify 2–3 wrong-type columns
# ==========================================
print("Activity 3: Possible wrong types (object but looks numeric/date)")

suspects = []
for col in df.columns:
    if df[col].dtype == "object":
        # Try numeric conversion
        numeric_conv = pd.to_numeric(df[col], errors="coerce")
        numeric_ratio = numeric_conv.notna().mean()

        # Try datetime conversion
        date_conv = pd.to_datetime(df[col], errors="coerce")
        date_ratio = date_conv.notna().mean()

        if numeric_ratio > 0.6:
            suspects.append((col, "Looks numeric but stored as text (object)"))
        elif date_ratio > 0.6:
            suspects.append((col, "Looks like date/time but stored as text (object)"))

suspects_df = pd.DataFrame(suspects, columns=["Column", "Why it looks wrong"])
print(suspects_df.head(10), "\n")


# ==========================================
# Activity 4: Missing values + key value checks
# ==========================================
print("Activity 4: Missing values")

missing_counts = df.isnull().sum().sort_values(ascending=False)
print("Missing counts (sorted):")
print(missing_counts, "\n")

print("Top 3 columns with most missing:")
print(missing_counts.head(3), "\n")

missing_percent = (df.isnull().sum() / len(df)) * 100
missing_table = pd.DataFrame({
    "Missing_Count": df.isnull().sum(),
    "Missing_Percent": missing_percent
}).sort_values("Missing_Percent", ascending=False)

print("Top 5 missingness table:")
print(missing_table.head(5), "\n")

# Optional: value_counts on key columns (if they exist)
sex_col = find_col(["sex", "gender"])
sev_col = find_col(["dengue_severity", "severity"])
hosp_col = find_col(["hospitalized", "hospitalised"])

for colname, label in [(sex_col, "SEX/GENDER"), (sev_col, "DENGUE_SEVERITY"), (hosp_col, "HOSPITALIZED")]:
    if colname:
        print(f"Value counts for {label} ({colname}):")
        print(df[colname].value_counts(dropna=False), "\n")


# ==========================================
# Activity 5: Duplicates and inconsistencies
# ==========================================
print("Activity 5: Duplicates")

print("Total duplicate rows:")
print(df.duplicated().sum(), "\n")

print("Duplicate rows (full-row duplicates):")
print(df[df.duplicated()], "\n")

# Extra challenge: patient_id duplicates
pid_col = find_col(["patient_id", "patientid", "id"])
if pid_col:
    print(f"Duplicate patient_id count (subset=['{pid_col}']):")
    print(df.duplicated(subset=[pid_col]).sum(), "\n")
else:
    print("No patient_id column found for subset duplicate check.\n")

print("Suggested rule (for your report): Keep the most complete or most recent record per patient_id.")
# Activity: Categorical inconsistencies (dengue_severity)

print("BEFORE unique dengue_severity:")
print(df["dengue_severity"].unique())

df["dengue_severity"] = (
    df["dengue_severity"]
    .replace({
        "Seveere": "Severe",
        "Modrate": "Moderate"
    })
    .astype(str)
    .str.strip()
    .str.title()
)

print("\nAFTER unique dengue_severity:")
print(df["dengue_severity"].unique())
# BEFORE counts
for col in ["hospitalized", "lab_confirmed"]:
    if col in df.columns:
        print(f"\nBEFORE counts for {col}:")
        print(df[col].value_counts(dropna=False))

# Normalize to exactly Yes/No
yes_set = {"yes", "y", "true", "1"}
no_set  = {"no", "n", "false", "0"}

def normalize_yes_no(x):
    if pd.isna(x):
        return x
    v = str(x).strip().lower()
    if v in yes_set:
        return "Yes"
    if v in no_set:
        return "No"
    return x  # keep anything unexpected

for col in ["hospitalized", "lab_confirmed"]:
    if col in df.columns:
        df[col] = df[col].apply(normalize_yes_no)

# AFTER counts
for col in ["hospitalized", "lab_confirmed"]:
    if col in df.columns:
        print(f"\nAFTER counts for {col}:")
        print(df[col].value_counts(dropna=False))
        # Activity 7: Convert data types and handle missing values

        import pandas as pd
        import numpy as np


        # --- Helper to find column names ignoring case/underscores/spaces
        def find_col(possible):
            norm = {c: c.lower().replace("_", "").replace(" ", "") for c in df.columns}
            targets = {p.lower().replace("_", "").replace(" ", "") for p in possible}
            for orig, n in norm.items():
                if n in targets:
                    return orig
            return None


        age_col = find_col(["age"])
        temp_col = find_col(["temperature_celsius", "temperaturecelsius", "temp_c", "temperature"])
        plat_col = find_col(["platelet_count", "plateletcount", "platelets"])

        # --- Show conversion success BEFORE cleaning (platelet_count)
        if plat_col:
            before_success = pd.to_numeric(df[plat_col], errors="coerce").notna().mean()
            print(f"Platelet numeric conversion success BEFORE cleaning: {before_success:.2%}")

        # --- Clean platelet_count: remove commas and handle 'k' suffix (e.g., '150k' -> 150000)
        if plat_col:
            s = df[plat_col].astype(str).str.strip().str.replace(",", "", regex=False)
            s = s.str.replace(r"(?i)\s*k$", "", regex=True)  # remove trailing 'k'
            # If original had 'k', multiply by 1000
            k_mask = df[plat_col].astype(str).str.contains(r"(?i)k\s*$", regex=True, na=False)
            nums = pd.to_numeric(s, errors="coerce")
            nums.loc[k_mask] = nums.loc[k_mask] * 1000
            df[plat_col] = nums

        # --- Convert age and temperature to numeric
        if age_col:
            df[age_col] = pd.to_numeric(df[age_col], errors="coerce")

        if temp_col:
            df[temp_col] = pd.to_numeric(df[temp_col], errors="coerce")

        # --- Show conversion success AFTER cleaning (platelet_count)
        if plat_col:
            after_success = df[plat_col].notna().mean()
            print(f"Platelet numeric conversion success AFTER cleaning:  {after_success:.2%}")

        # --- Fill missing age with median
        if age_col:
            median_age = df[age_col].median()
            df[age_col].fillna(median_age, inplace=True)
            print(f"Filled missing {age_col} with median: {median_age}")

        # --- Drop duplicate rows
        before_dupes = df.duplicated().sum()
        df.drop_duplicates(inplace=True)
        after_dupes = df.duplicated().sum()
        print(f"Dropped duplicates: before={before_dupes}, after={after_dupes}")
        # =========================
        # Activity 8: Outlier detection
        # =========================

        import numpy as np

        print("\nActivity 8: Outlier detection")

        # 1) Inspect extremes
        print("\nSummary stats (numeric):")
        print(df.describe())


        # Helper to find column names if slightly different
        def find_col(possible):
            norm = {c: c.lower().replace("_", "").replace(" ", "") for c in df.columns}
            targets = {p.lower().replace("_", "").replace(" ", "") for p in possible}
            for orig, n in norm.items():
                if n in targets:
                    return orig
            return None


        plat_col = find_col(["platelet_count", "plateletcount", "platelets"])
        temp_col = find_col(["temperature_celsius", "temperaturecelsius", "temp_c", "temperature"])


        # Function to cap using IQR
        def cap_iqr(series):
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            capped = series.clip(lower=lower, upper=upper)
            return capped, lower, upper


        # 2) IQR capping for platelet_count
        if plat_col:
            before_min, before_max = df[plat_col].min(), df[plat_col].max()
            df[plat_col], p_low, p_high = cap_iqr(df[plat_col])
            after_min, after_max = df[plat_col].min(), df[plat_col].max()
            print(f"\nIQR caps for {plat_col}: lower={p_low:.2f}, upper={p_high:.2f}")
            print(f"{plat_col} before min/max: {before_min} / {before_max}")
            print(f"{plat_col} after  min/max: {after_min} / {after_max}")

            # Optional visual
            try:
                df[plat_col].plot(kind="box", title=f"{plat_col} (after IQR capping)")
            except Exception as e:
                print("Plot skipped:", e)

        # 3) IQR capping for temperature_celsius
        if temp_col:
            before_min, before_max = df[temp_col].min(), df[temp_col].max()
            df[temp_col], t_low, t_high = cap_iqr(df[temp_col])
            after_min, after_max = df[temp_col].min(), df[temp_col].max()
            print(f"\nIQR caps for {temp_col}: lower={t_low:.2f}, upper={t_high:.2f}")
            print(f"{temp_col} before min/max: {before_min} / {before_max}")
            print(f"{temp_col} after  min/max: {after_min} / {after_max}")

        # 4) Compare with simple clinical caps
        # Example clinical caps: platelets [20,000, 600,000]; temperature [35, 45] °C
        if plat_col:
            clinical_low_p, clinical_high_p = 20000, 600000
            clipped_p = df[plat_col].clip(lower=clinical_low_p, upper=clinical_high_p)
            print(f"\nClinical caps for {plat_col}: [{clinical_low_p}, {clinical_high_p}]")
            print("Example after clinical capping (min/max):",
                  clipped_p.min(), "/", clipped_p.max())

        if temp_col:
            clinical_low_t, clinical_high_t = 35, 45
            clipped_t = df[temp_col].clip(lower=clinical_low_t, upper=clinical_high_t)
            print(f"Clinical caps for {temp_col}: [{clinical_low_t}, {clinical_high_t}]")
            print("Example after clinical capping (min/max):",
                  clipped_t.min(), "/", clipped_t.max())

        print("\nOne-sentence note (for report):")
        print("Clinical caps are often more defensible in health data because they reflect physiological limits, "
              "while IQR caps depend on the sample distribution and may clip valid extreme cases during outbreaks.")
        # =========================
        # Activity 9: Validate and summarize
        # =========================

        print("\nActivity 9: Validate and summarize")

        # 1) Compare missing values before vs after cleaning
        print("\nMissing values BEFORE cleaning (raw):")
        print(raw.isnull().sum())

        print("\nMissing values AFTER cleaning (df):")
        print(df.isnull().sum())

        # 2) Confirm types after cleaning
        print("\nData types AFTER cleaning (df.info()):")
        df.info()

        # 3) Short summary (3–4 actions) - print so you can copy into your report
        summary = [
            "Standardised categorical labels (e.g., dengue_severity, hospitalized, lab_confirmed) to reduce inconsistent values and improve analysis reliability.",
            "Converted platelet_count (and other numeric fields where needed) to numeric types to enable correct summary statistics and modelling.",
            "Removed duplicate rows to avoid repeating the same record and biasing results.",
            "Created a raw backup copy (raw = df.copy()) to preserve the original dataset before cleaning."
        ]
        print("\nCLEANING SUMMARY (copy to report):")
        for i, s in enumerate(summary, 1):
            print(f"{i}. {s}")

        # 4) Extra challenge: tiny audit dict (before/after counts)
        audit = {
            "rows_before": len(raw),
            "rows_after": len(df),
            "full_row_duplicates_before": raw.duplicated().sum(),
            "full_row_duplicates_after": df.duplicated().sum(),
        }

        # Optional: audit for one or two columns (if they exist)
        if "hospitalized" in raw.columns and "hospitalized" in df.columns:
            audit["hospitalized_counts_before"] = raw["hospitalized"].value_counts(dropna=False).to_dict()
            audit["hospitalized_counts_after"] = df["hospitalized"].value_counts(dropna=False).to_dict()

        if "dengue_severity" in raw.columns and "dengue_severity" in df.columns:
            audit["severity_counts_before"] = raw["dengue_severity"].value_counts(dropna=False).to_dict()
            audit["severity_counts_after"] = df["dengue_severity"].value_counts(dropna=False).to_dict()

        print("\nAUDIT DICT:")
        print(audit)

