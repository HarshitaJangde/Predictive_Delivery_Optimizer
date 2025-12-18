# Predictive Delivery Optimizer with Cost Intelligence

## Introduction
The Predictive Delivery Optimizer is a Python & Streamlit-based interactive web application designed to transform logistics operations from reactive to proactive.
It predicts potential delivery delays, estimates financial impact, and provides actionable recommendations for each order, helping logistics companies optimize operations, reduce costs, and improve customer satisfaction.

This project merges multiple datasets from the logistics ecosystem to deliver data-driven insights and visualizations.

---

## Features

1. [Features](#features)
2. [Datasets Used](#datasets-used)
3. [Installation](#installation)
4. [Running the Application](#running-the-application)
5. [Application Sections](#application-sections)
6. [Downloadable CSV](#downloadable-csv)
7. [Acknowledgement](#acknowledgement)

---

## Features

* Predicts delivery delay probability for each order.
* Classifies orders into **High, Medium, Low Risk** categories.
* Estimates **financial impact** of delayed deliveries.
* Provides actionable **recommendations** to mitigate delay.
* Interactive dashboards with **filters and selections**.
* Visualizations to explore trends, costs, and operational insights.
* Ability to **export filtered insights** for further analysis.

---

## Datasets Used

* `orders.csv` – Order-level information: IDs, priority, value, product category.
* `delivery_performance.csv` – Delivery execution: promised vs actual times, delivery status.
* `routes_distance.csv` – Route metrics: distance, traffic, weather impact.
* `cost_breakdown.csv` – Detailed cost components: fuel, labor, maintenance.

> These datasets are merged and processed to create predictive features and derived metrics.

---

## Installation

1. Download the code & CSVs (OR Clone the repository):

```bash
git clone https://github.com/HarshitaJangde/Predictive_Delivery_Optimizer.git
```

2. Create a virtual environment :
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

3. Install required packages:

``` bash
pip install -r requirements.txt
```

4. Run the Application:
```bash
streamlit run app.py
```

The app will open in your default web browser.

Ensure that all CSV files (orders.csv, delivery_performance.csv, routes_distance.csv, cost_breakdown.csv) are inside Data folder which is in the same folder as app.py.

---

## Application Sections

### 1. Executive Overview

* Displays **key KPIs**:

  * Delay rate
  * Average delivery cost
  * Total orders
* Gives a high-level understanding of logistics performance.

### 2. Operational Insights (Visualizations)

* **Bar Chart** – Delays by priority.
* **Pie Chart** – On-time vs delayed deliveries.
* **Scatter Plot** – Distance vs traffic delay impact.
* **Line Chart** – Average delivery cost over time.
* Interactive charts help explore trends and operational challenges.

### 3. Order-Level Prediction

* Select a specific **Order ID** to predict delay probability.
* Displays:

  * Delay Probability
  * Risk Level (Low / Medium / High)
  * Estimated Cost Impact
  * Recommended corrective action

### 4. Export Insights

* Allows users to **download filtered data** for further analysis or reporting.
* The exported CSV includes:

  * Order details
  * Derived metrics (delay risk, total cost)
  * Prediction outputs
  * Risk classification
  * Recommended actions

## Downloadable CSV

* **File name:** `predictive_delivery_insights.csv`
* **Content:**  

  * Filtered orders based on selected priority or criteria
  * Predicted delay probability for each order
  * Estimated financial impact
  * Recommended corrective action

> This allows business teams to make data-driven operational decisions.

---

## Acknowledgement

Thank you to the reviewers and recruiters for taking the time to evaluate this project.
I hope this project demonstrates both technical proficiency and business impact, and I look forward to the opportunity to contribute to innovative logistics solutions with your team in the future.