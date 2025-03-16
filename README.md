# DSCI_532_individual-assignment_jc24907

## Overview
This dashboard application was developed as part of the DSCI 532 course at UBC. It aims to provide an interactive and insightful visualization platform for exploring and analyzing data.

## Motivation
Target audience: Climate policy makers and environmental researchers

Climate change is one of the most pressing challenges of our time, with CO2 emissions being a major contributor to global temperature increases. Policy makers and researchers need tools to understand and predict how changes in different types of emissions affect temperature trajectories. This dashboard enables users to explore historical temperature data for G7 countries and simulate future temperature increases based on adjustments to gas, coal, and oil CO2 emissions. The predictions are made using a machine learning model. By providing an interactive way to visualize these relationships, the dashboard helps inform evidence-based policy decisions and enhance understanding of emission impacts.

## App Description
The dashboard provides an interactive platform for analyzing temperature increases and CO2 emissions with the following features:

- Data source for historical emissions and temperature increase: https://github.com/owid/co2-data
- Historical temperature data visualization for G7 countries (solid lines)
- Future temperature predictions based on adjustable CO2 parameters (dotted lines)
- Interactive controls for modifying gas, coal, and oil CO2 output (±15%)
- Country selection filter for focused analysis
- Ridge linear regression model for temperature forecasting

## Installation Instructions

### Prerequisites
- Python 3.8 or higher
- conda

### Setup Steps
1. Clone the repository:
   ```bash
   git clone https://github.ubc.ca/mds-2024-25/DSCI_532_individual-assignment_jc24907
   cd DSCI_532_individual-assignment_jc24907
   ```

2. Create and activate a virtual environment:
   ```bash
   conda env create -n dashboard -f environment.yml
   conda activate dashboard
   ```

3. Run the application:
   ```bash
   streamlit run src/app.py
   ```