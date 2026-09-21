from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from analysis import (
    get_overview,
    get_countries,
    get_country,
    get_country_trend,
    get_global_trend,
    get_rankings,
    get_highest_mmr_countries,
    get_lowest_mmr_countries,
    get_sdg_progress,
    get_country_sdg_progress,
    get_yearly_statistics,
    get_dataset_summary
)

from prediction import (
    predict_mmr,
    predict_until_2030,
    get_prediction_chart_data,
    evaluate_model,
    get_complete_prediction
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Maternal Health SDG 3.1 API",
    description=(
        "Backend API for tracking Maternal Mortality "
        "Ratio and progress toward SDG 3.1."
    ),
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Maternal Health SDG 3.1 API is running",
        "status": "success",
        "version": "1.0.0"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "Backend is running correctly"
    }


# ============================================================
# OVERVIEW
# ============================================================

@app.get("/api/overview")
def overview():

    try:

        return get_overview()

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Error loading overview: {str(e)}"
        )


# ============================================================
# AREAS / COUNTRIES
# ============================================================

@app.get("/api/countries")
def countries():

    try:

        return get_countries()

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Error loading areas: {str(e)}"
        )


# ============================================================
# SINGLE AREA DETAILS
# ============================================================

@app.get("/api/country/{country}")
def country_details(country: str):

    try:

        result = get_country(country)

        if result is None:

            raise HTTPException(
                status_code=404,
                detail=f"Area '{country}' not found"
            )

        return result

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Error loading area data: {str(e)}"
        )


# ============================================================
# SINGLE AREA TREND
# ============================================================

@app.get("/api/country/{country}/trend")
def country_trend(country: str):

    try:

        result = get_country_trend(country)

        if not result:

            raise HTTPException(
                status_code=404,
                detail=f"No trend data found for '{country}'"
            )

        return result

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Error loading area trend: {str(e)}"
        )


# ============================================================
# GLOBAL / OVERALL TREND
# ============================================================

@app.get("/api/trend")
def global_trend():

    try:

        return get_global_trend()

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Error loading trend: {str(e)}"
        )


# ============================================================
# RANKINGS
# ============================================================

@app.get("/api/rankings")
def rankings(limit: int = 10):

    try:

        if limit < 1:
            limit = 10

        if limit > 100:
            limit = 100

        return get_rankings(limit)

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Error loading rankings: {str(e)}"
        )


# ============================================================
# HIGHEST MMR
# ============================================================

@app.get("/api/rankings/highest")
def highest_mmr(limit: int = 10):

    try:

        if limit < 1:
            limit = 10

        if limit > 100:
            limit = 100

        return get_highest_mmr_countries(limit)

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Error loading highest MMR areas: {str(e)}"
        )


# ============================================================
# LOWEST MMR
# ============================================================

@app.get("/api/rankings/lowest")
def lowest_mmr(limit: int = 10):

    try:

        if limit < 1:
            limit = 10

        if limit > 100:
            limit = 100

        return get_lowest_mmr_countries(limit)

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Error loading lowest MMR areas: {str(e)}"
        )


# ============================================================
# SDG PROGRESS
# ============================================================

@app.get("/api/sdg-progress")
def sdg_progress():

    try:

        return get_sdg_progress()

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Error loading SDG progress: {str(e)}"
        )


# ============================================================
# SDG PROGRESS FOR ONE AREA
# ============================================================

@app.get("/api/sdg-progress/{country}")
def country_sdg_progress(country: str):

    try:

        result = get_country_sdg_progress(country)

        if result is None:

            raise HTTPException(
                status_code=404,
                detail=(
                    f"No SDG progress data found "
                    f"for '{country}'"
                )
            )

        return result

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Error loading SDG progress: {str(e)}"
        )


# ============================================================
# YEARLY STATISTICS
# ============================================================

@app.get("/api/statistics/yearly")
def yearly_statistics():

    try:

        return get_yearly_statistics()

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Error loading yearly statistics: "
                f"{str(e)}"
            )
        )


# ============================================================
# DATASET SUMMARY
# ============================================================

@app.get("/api/dataset-summary")
def dataset_summary():

    try:

        return get_dataset_summary()

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Error loading dataset summary: "
                f"{str(e)}"
            )
        )


# ============================================================
# PREDICTION FOR SPECIFIC YEAR
# ============================================================

@app.get("/api/prediction/{country}/{year}")
def prediction_for_year(
    country: str,
    year: int
):

    try:

        if year < 2024:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Prediction year must be "
                    "2024 or later."
                )
            )

        if year > 2050:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Prediction year must not "
                    "be greater than 2050."
                )
            )

        result = predict_mmr(
            country=country,
            target_year=year
        )

        if result is None:

            raise HTTPException(
                status_code=404,
                detail=(
                    f"Prediction could not be generated "
                    f"for '{country}'. "
                    f"At least two historical MMR "
                    f"records may be required."
                )
            )

        return result

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Error generating prediction: "
                f"{str(e)}"
            )
        )


# ============================================================
# COMPLETE 2030 PREDICTION
# ============================================================

@app.get("/api/prediction/{country}")
def prediction(country: str):

    try:

        result = get_complete_prediction(
            country
        )

        if result is None:

            raise HTTPException(
                status_code=404,
                detail=(
                    f"Prediction could not be generated "
                    f"for '{country}'. "
                    f"At least two historical MMR "
                    f"records may be required."
                )
            )

        return result

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Error generating prediction: "
                f"{str(e)}"
            )
        )


# ============================================================
# FORECAST DATA
# ============================================================

@app.get("/api/prediction/{country}/forecast")
def prediction_forecast(country: str):

    try:

        result = predict_until_2030(
            country
        )

        if result is None:

            raise HTTPException(
                status_code=404,
                detail=(
                    f"No forecast available "
                    f"for '{country}'"
                )
            )

        return {
            "country": country,
            "forecast": result
        }

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Error generating forecast: "
                f"{str(e)}"
            )
        )


# ============================================================
# PREDICTION CHART DATA
# ============================================================

@app.get("/api/prediction/{country}/chart")
def prediction_chart(country: str):

    try:

        result = get_prediction_chart_data(
            country
        )

        if result is None:

            raise HTTPException(
                status_code=404,
                detail=(
                    f"No prediction chart data "
                    f"found for '{country}'"
                )
            )

        return result

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Error loading prediction chart: "
                f"{str(e)}"
            )
        )


# ============================================================
# MODEL EVALUATION
# ============================================================

@app.get("/api/prediction/{country}/evaluation")
def prediction_evaluation(country: str):

    try:

        result = evaluate_model(
            country
        )

        if result is None:

            raise HTTPException(
                status_code=404,
                detail=(
                    f"Model evaluation could not "
                    f"be performed for '{country}'"
                )
            )

        return result

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Error evaluating prediction model: "
                f"{str(e)}"
            )
        )


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )