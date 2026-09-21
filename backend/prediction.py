import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "sdg_maternal_health.csv"
)


# ============================================================
# LOAD MMR DATA
# ============================================================

def load_mmr_data():

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

    # Extract year from TimePeriod
    if "TimePeriod" in df.columns:

        df["Year"] = pd.to_numeric(
            df["TimePeriod"]
            .astype(str)
            .str.extract(r"(\d{4})")[0],
            errors="coerce"
        )

    elif "Year" not in df.columns:

        raise ValueError(
            "Year information not found in dataset"
        )

    # Convert DataValue
    df["DataValue"] = pd.to_numeric(
        df["DataValue"],
        errors="coerce"
    )

    # ONLY indicator 3.1.1
    df = df[
        df["Indicator"]
        .astype(str)
        .str.strip()
        .str.startswith("3.1.1:")
    ].copy()

    df = df.dropna(
        subset=[
            "AreaName",
            "Year",
            "DataValue"
        ]
    )

    return df


# ============================================================
# PREPARE AREA DATA
# ============================================================

def prepare_country_data(country):

    df = load_mmr_data()

    result = df[
        df["AreaName"]
        .astype(str)
        .str.strip()
        .str.lower()
        ==
        str(country).strip().lower()
    ].copy()

    if result.empty:
        return pd.DataFrame()

    result = (
        result
        .groupby("Year", as_index=False)["DataValue"]
        .mean()
        .rename(columns={"DataValue": "MMR"})
        .sort_values("Year")
    )

    return result


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(country):

    df = prepare_country_data(country)

    if df.empty:
        return None

    if len(df) < 2:
        return None

    X = df[["Year"]]
    y = df["MMR"]

    model = LinearRegression()

    model.fit(X, y)

    return model, df


# ============================================================
# PREDICT ONE YEAR
# ============================================================

def predict_mmr(country, target_year=2030):

    trained = train_model(country)

    if trained is None:
        return None

    model, df = trained

    prediction = model.predict(
        [[target_year]]
    )[0]

    prediction = max(0, float(prediction))

    return {
        "success": True,
        "country": country,
        "target_year": int(target_year),
        "prediction": round(prediction, 2),
        "predicted_mmr": round(prediction, 2)
    }


# ============================================================
# PREDICTION STATUS
# ============================================================

def prediction_status(mmr):

    if mmr < 70:
        return "Target Achieved"

    elif mmr <= 100:
        return "Near Target"

    elif mmr <= 200:
        return "At Risk"

    else:
        return "Off Track"


# ============================================================
# PREDICT UNTIL 2030
# ============================================================

def predict_until_2030(country):

    trained = train_model(country)

    if trained is None:
        return []

    model, df = trained

    latest_year = int(df["Year"].max())

    start_year = max(
        latest_year + 1,
        2024
    )

    if start_year > 2030:
        return []

    years = list(
        range(
            start_year,
            2031
        )
    )

    predictions = []

    for year in years:

        value = model.predict(
            [[year]]
        )[0]

        value = max(
            0,
            float(value)
        )

        predictions.append({
            "Year": year,
            "MMR": round(value, 2),
            "prediction": round(value, 2)
        })

    return predictions


# ============================================================
# CHART DATA
# ============================================================

def get_prediction_chart_data(country):

    df = prepare_country_data(country)

    if df.empty:
        return None

    historical = []

    for _, row in df.iterrows():

        historical.append({
            "Year": int(row["Year"]),
            "MMR": round(
                float(row["MMR"]),
                2
            )
        })

    forecast = predict_until_2030(country)

    return {
        "historical": historical,
        "forecast": forecast
    }


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(country):

    trained = train_model(country)

    if trained is None:
        return None

    model, df = trained

    X = df[["Year"]]
    y = df["MMR"]

    predictions = model.predict(X)

    mae = mean_absolute_error(
        y,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y,
            predictions
        )
    )

    r2 = r2_score(
        y,
        predictions
    )

    return {
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "r2": round(float(r2), 4)
    }


# ============================================================
# COMPLETE PREDICTION
# ============================================================

def get_complete_prediction(country):

    result = predict_mmr(
        country,
        2030
    )

    if result is None:
        return None

    prediction = result["prediction"]

    status = prediction_status(
        prediction
    )

    chart_data = get_prediction_chart_data(
        country
    )

    evaluation = evaluate_model(
        country
    )

    return {
        "success": True,

        "country": country,

        "target_year": 2030,

        "prediction": prediction,

        "predicted_mmr": prediction,

        "prediction_2030": prediction,

        "target": 70,

        "status": status,

        "historical": chart_data["historical"]
        if chart_data else [],

        "forecast": chart_data["forecast"]
        if chart_data else [],

        "evaluation": evaluation
    }


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print("\n===================================")
    print("MATERNAL HEALTH PREDICTION TEST")
    print("===================================\n")

    try:

        df = load_mmr_data()

        print(
            f"MMR records loaded: {len(df)}"
        )

        print(
            f"Areas available: "
            f"{df['AreaName'].nunique()}"
        )

        areas = (
            df["AreaName"]
            .dropna()
            .unique()
        )

        if len(areas) > 0:

            test_area = areas[0]

            print(
                f"\nTesting area: {test_area}"
            )

            result = get_complete_prediction(
                test_area
            )

            print("\nPrediction result:")

            print(result)

        else:

            print(
                "No areas found."
            )

    except Exception as e:

        print(
            "\nERROR:",
            str(e)
        )