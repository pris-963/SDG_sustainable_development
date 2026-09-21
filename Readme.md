# Tracking Maternal Health Progress Toward SDG 3.1

A data analysis and machine learning project that tracks **Maternal Mortality Ratio (MMR)** and evaluates progress toward **United Nations Sustainable Development Goal 3.1**.

The project provides a FastAPI backend for processing maternal health data, performing statistical analysis, generating predictions, and serving results to a simple web dashboard.

---

## 🎯 Project Objective

The main objective is to analyze maternal health indicators and track progress toward **SDG Target 3.1**, which aims to:

> Reduce the global maternal mortality ratio to less than 70 deaths per 100,000 live births by 2030.

The system focuses primarily on:

- Maternal Mortality Ratio (MMR)
- Skilled birth attendance
- Antenatal care coverage
- SDG 3.1 progress
- Historical MMR trends
- Area-wise MMR comparison
- MMR prediction using Machine Learning

---

## 📌 Problem Statement

Maternal mortality remains an important public health challenge. Understanding historical maternal mortality trends and identifying whether regions are progressing toward the SDG 3.1 target can help support data-driven analysis.

This project analyzes available maternal health data, visualizes MMR trends, compares areas, measures progress toward the 2030 target, and uses machine learning to estimate future MMR values.

---

## 💡 Proposed Solution

The system provides a web-based dashboard connected to a Python FastAPI backend.

The backend:

1. Loads the maternal health dataset.
2. Cleans and processes the data.
3. Extracts Maternal Mortality Ratio records.
4. Calculates statistical summaries.
5. Generates area-wise and yearly trends.
6. Measures progress toward the SDG 3.1 target.
7. Ranks areas based on MMR.
8. Uses Linear Regression to estimate future MMR.
9. Provides the processed results through REST APIs.

The frontend displays these results through interactive charts, KPI cards, tables, and progress indicators.

---

## 🧠 Machine Learning Approach

### Linear Regression

Linear Regression is used to estimate future Maternal Mortality Ratio based on historical yearly MMR observations.

### Input

- Year

### Output

- Predicted Maternal Mortality Ratio

### Example

Historical data:

```text
Year       MMR
2015       130
2016       125
2017       120
2018       115