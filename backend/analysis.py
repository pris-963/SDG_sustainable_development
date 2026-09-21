
import os
import re
import pandas as pd
import numpy as np


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "sdg_maternal_health.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Extract Year
    # --------------------------------------------------------

    if "TimePeriod" in df.columns:

        df["Year"] = pd.to_numeric(
            df["TimePeriod"]
            .astype(str)
            .str.extract(r"(\d{4})")[0],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Convert DataValue
    # --------------------------------------------------------

    if "DataValue" in df.columns:

        df["DataValue"] = (
            df["DataValue"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
        )

        df["DataValue"] = pd.to_numeric(
            df["DataValue"],
            errors="coerce"
        )

    return df


# ============================================================
# MATERNAL HEALTH DATA
# ============================================================

def get_maternal_data():

    df = load_data()

    if "Indicator" not in df.columns:
        return pd.DataFrame()

    return df[
        df["Indicator"]
        .astype(str)
        .str.strip()
        .str.startswith("3.1")
    ].copy()


# ============================================================
# MMR DATA
# SDG INDICATOR 3.1.1
# ============================================================

def get_mmr_data():

    df = load_data()

    if "Indicator" not in df.columns:
        return pd.DataFrame()

    result = df[
        df["Indicator"]
        .astype(str)
        .str.strip()
        .str.startswith("3.1.1:")
    ].copy()

    result = result.dropna(
        subset=[
            "AreaName",
            "Year",
            "DataValue"
        ]
    )

    return result


# ============================================================
# STATUS
# ============================================================

def get_sdg_status(mmr):

    if mmr is None or pd.isna(mmr):
        return "Unknown"

    mmr = float(mmr)

    if mmr < 70:
        return "Target Achieved"

    elif mmr <= 100:
        return "Near Target"

    elif mmr <= 200:
        return "At Risk"

    else:
        return "Off Track"


# ============================================================
# OVERVIEW
# ============================================================

def get_overview():

    df = get_mmr_data()

    if df.empty:

        return {
            "success": True,
            "average_mmr": None,
            "avg_mmr": None,
            "latest_year": None,
            "area_count": 0,
            "total_areas": 0,
            "target": 70
        }

    # Average MMR
    average_mmr = float(
        df["DataValue"].mean()
    )

    # Latest year
    latest_year = int(
        df["Year"].max()
    )

    # Number of areas
    area_count = int(
        df["AreaName"]
        .nunique()
    )

    return {

        "success": True,

        "average_mmr": round(
            average_mmr,
            2
        ),

        "avg_mmr": round(
            average_mmr,
            2
        ),

        "latest_year": latest_year,

        "area_count": area_count,

        "total_areas": area_count,

        "target": 70
    }


# ============================================================
# AREAS
# ============================================================

def get_countries():

    df = get_mmr_data()

    if df.empty:
        return []

    areas = (
        df["AreaName"]
        .dropna()
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    return areas


# ============================================================
# SINGLE AREA DETAILS
# ============================================================

def get_country(country):

    df = get_mmr_data()

    if df.empty:
        return None

    result = df[
        df["AreaName"]
        .astype(str)
        .str.strip()
        .str.lower()
        ==
        str(country).strip().lower()
    ].copy()

    if result.empty:
        return None

    result = (
        result
        .groupby("Year", as_index=False)["DataValue"]
        .mean()
        .sort_values("Year")
    )

    latest = result.iloc[-1]

    mmr = float(
        latest["DataValue"]
    )

    year = int(
        latest["Year"]
    )

    return {

        "success": True,

        "country": country,

        "area": country,

        "AreaName": country,

        "Year": year,

        "year": year,

        "MMR": round(
            mmr,
            2
        ),

        "mmr": round(
            mmr,
            2
        ),

        "DataValue": round(
            mmr,
            2
        ),

        "status": get_sdg_status(
            mmr
        ),

        "target": 70
    }


# ============================================================
# AREA TREND
# ============================================================

def get_country_trend(country):

    df = get_mmr_data()

    if df.empty:
        return []

    result = df[
        df["AreaName"]
        .astype(str)
        .str.strip()
        .str.lower()
        ==
        str(country).strip().lower()
    ].copy()

    if result.empty:
        return []

    result = (
        result
        .groupby("Year", as_index=False)["DataValue"]
        .mean()
        .sort_values("Year")
    )

    output = []

    for _, row in result.iterrows():

        year = int(
            row["Year"]
        )

        mmr = float(
            row["DataValue"]
        )

        output.append({

            "Year": year,

            "year": year,

            "MMR": round(
                mmr,
                2
            ),

            "mmr": round(
                mmr,
                2
            ),

            "DataValue": round(
                mmr,
                2
            )

        })

    return output


# ============================================================
# GLOBAL / OVERALL TREND
# ============================================================

def get_global_trend():

    df = get_mmr_data()

    if df.empty:
        return []

    result = (
        df
        .groupby("Year", as_index=False)["DataValue"]
        .mean()
        .sort_values("Year")
    )

    output = []

    for _, row in result.iterrows():

        year = int(
            row["Year"]
        )

        mmr = float(
            row["DataValue"]
        )

        output.append({

            "Year": year,

            "year": year,

            "MMR": round(
                mmr,
                2
            ),

            "mmr": round(
                mmr,
                2
            ),

            "average_mmr": round(
                mmr,
                2
            )

        })

    return output


# ============================================================
# AREA RANKINGS
# ============================================================

def get_rankings(limit=10):

    df = get_mmr_data()

    if df.empty:
        return []

    # Get latest observation for every area
    latest_year = (
        df.groupby("AreaName")["Year"]
        .max()
        .reset_index()
    )

    latest = df.merge(
        latest_year,
        on=[
            "AreaName",
            "Year"
        ],
        how="inner"
    )

    # If duplicate observations exist for
    # same area/year, average them.
    latest = (
        latest
        .groupby(
            [
                "AreaName",
                "Year"
            ],
            as_index=False
        )["DataValue"]
        .mean()
    )

    # Sort by MMR
    latest = latest.sort_values(
        "DataValue",
        ascending=True
    )

    output = []

    for index, row in latest.head(limit).iterrows():

        area = str(
            row["AreaName"]
        )

        year = int(
            row["Year"]
        )

        mmr = float(
            row["DataValue"]
        )

        output.append({

            "rank": len(output) + 1,

            "AreaName": area,

            "area": area,

            "country": area,

            "Year": year,

            "year": year,

            "MMR": round(
                mmr,
                2
            ),

            "mmr": round(
                mmr,
                2
            ),

            "DataValue": round(
                mmr,
                2
            ),

            "status": get_sdg_status(
                mmr
            ),

            "target": 70

        })

    return output


# ============================================================
# HIGHEST MMR
# ============================================================

def get_highest_mmr_countries(limit=10):

    rankings = get_rankings(
        limit=1000
    )

    rankings = sorted(
        rankings,
        key=lambda x: x["MMR"],
        reverse=True
    )

    return rankings[:limit]


# ============================================================
# LOWEST MMR
# ============================================================

def get_lowest_mmr_countries(limit=10):

    rankings = get_rankings(
        limit=1000
    )

    return rankings[:limit]


# ============================================================
# SDG PROGRESS - ALL AREAS
# ============================================================

def get_sdg_progress():

    df = get_mmr_data()

    if df.empty:
        return []

    output = []

    latest_years = (
        df.groupby("AreaName")["Year"]
        .max()
        .reset_index()
    )

    latest = df.merge(
        latest_years,
        on=[
            "AreaName",
            "Year"
        ],
        how="inner"
    )

    latest = (
        latest
        .groupby(
            [
                "AreaName",
                "Year"
            ],
            as_index=False
        )["DataValue"]
        .mean()
    )

    for _, row in latest.iterrows():

        area = str(
            row["AreaName"]
        )

        mmr = float(
            row["DataValue"]
        )

        year = int(
            row["Year"]
        )

        if mmr <= 70:

            progress = 100.0

        else:

            progress = (
                70 / mmr
            ) * 100

        progress = max(
            0,
            min(
                100,
                progress
            )
        )

        output.append({

            "AreaName": area,

            "area": area,

            "country": area,

            "Year": year,

            "year": year,

            "MMR": round(
                mmr,
                2
            ),

            "current_mmr": round(
                mmr,
                2
            ),

            "target": 70,

            "progress": round(
                progress,
                2
            ),

            "progress_percentage": round(
                progress,
                2
            ),

            "status": get_sdg_status(
                mmr
            )

        })

    return output


# ============================================================
# SDG PROGRESS - SINGLE AREA
# ============================================================

def get_country_sdg_progress(country):

    df = get_mmr_data()

    if df.empty:
        return None

    result = df[
        df["AreaName"]
        .astype(str)
        .str.strip()
        .str.lower()
        ==
        str(country).strip().lower()
    ].copy()

    if result.empty:
        return None

    result = (
        result
        .groupby(
            "Year",
            as_index=False
        )["DataValue"]
        .mean()
        .sort_values("Year")
    )

    latest = result.iloc[-1]

    year = int(
        latest["Year"]
    )

    current_mmr = float(
        latest["DataValue"]
    )

    target = 70.0

    if current_mmr <= target:

        progress = 100.0

    else:

        progress = (
            target / current_mmr
        ) * 100

    progress = max(
        0,
        min(
            100,
            progress
        )
    )

    return {

        "success": True,

        "country": country,

        "area": country,

        "AreaName": country,

        "Year": year,

        "year": year,

        "MMR": round(
            current_mmr,
            2
        ),

        "mmr": round(
            current_mmr,
            2
        ),

        "current_mmr": round(
            current_mmr,
            2
        ),

        "target": 70,

        "progress": round(
            progress,
            2
        ),

        "progress_percentage": round(
            progress,
            2
        ),

        "status": get_sdg_status(
            current_mmr
        )
    }


# ============================================================
# YEARLY STATISTICS
# ============================================================

def get_yearly_statistics():

    df = get_mmr_data()

    if df.empty:
        return []

    result = (
        df
        .groupby("Year")["DataValue"]
        .agg(
            average="mean",
            minimum="min",
            maximum="max",
            count="count"
        )
        .reset_index()
        .sort_values("Year")
    )

    return result.to_dict(
        orient="records"
    )


# ============================================================
# CORRELATION
# ============================================================

def get_correlation():

    df = get_maternal_data()

    if df.empty:
        return {}

    numeric = df.select_dtypes(
        include=np.number
    )

    if numeric.empty:
        return {}

    return numeric.corr().round(
        3
    ).to_dict()


# ============================================================
# DATASET SUMMARY
# ============================================================

def get_dataset_summary():

    df = load_data()

    return {

        "rows": int(
            len(df)
        ),

        "columns": int(
            len(df.columns)
        ),

        "areas": int(
            df["AreaName"]
            .nunique()
        )
        if "AreaName" in df.columns
        else 0,

        "indicators": int(
            df["Indicator"]
            .nunique()
        )
        if "Indicator" in df.columns
        else 0,

        "years": (
            sorted(
                df["Year"]
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )
            if "Year" in df.columns
            else []
        )

    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("MATERNAL HEALTH ANALYSIS TEST")
    print("========================================")

    try:

        df = load_data()

        print(
            f"\nTotal dataset rows: {len(df)}"
        )

        mmr = get_mmr_data()

        print(
            f"MMR rows: {len(mmr)}"
        )

        print(
            f"Areas: {mmr['AreaName'].nunique()}"
        )

        print(
            f"Years: {sorted(mmr['Year'].unique())}"
        )

        print(
            "\nOverview:"
        )

        print(
            get_overview()
        )

        areas = get_countries()

        print(
            "\nFirst 5 areas:"
        )

        print(
            areas[:5]
        )

        if areas:

            test_area = areas[0]

            print(
                f"\nTesting area: {test_area}"
            )

            print(
                "\nDetails:"
            )

            print(
                get_country(test_area)
            )

            print(
                "\nSDG Progress:"
            )

            print(
                get_country_sdg_progress(
                    test_area
                )
            )

            print(
                "\nTrend:"
            )

            print(
                get_country_trend(
                    test_area
                )
            )

            print(
                "\nRankings:"
            )

            print(
                get_rankings(5)
            )

    except Exception as e:

        print(
            "\nERROR:",
            str(e)
        )

