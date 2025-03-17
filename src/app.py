import streamlit as st
import pandas as pd
import altair as alt
import pickle

# Load data and model
df = pd.read_csv('data/processed/temperature.csv')
with open("notebook/model.pkl", "rb") as file:
    model = pickle.load(file)

# Sidebar for filters
st.sidebar.title('Filters')

# Add checkboxes to select which countries to display
countries = df['country'].unique()
selected_countries = st.sidebar.multiselect('Select countries to display', countries, default=countries)

# Add checkbox to toggle predictions visibility
show_predictions = st.sidebar.checkbox('Show predictions', value=True)

# Extract 2022 values for each column
g7 = ['Canada', 'France', 'Germany', 'Italy', 'Japan', 'United Kingdom', 'United States']
baseline = df[(df['year'] == 2022) & (df['country'].isin(g7))]

st.sidebar.markdown("Adjust annual CO2 output changes for gas, coal, and oil (±15%)")

# Add sliders for continuous columns with relative values
gas_value = st.sidebar.slider('Gas CO2 output (%)', -15, 15, 0)
coal_value = st.sidebar.slider('Coal CO2 output (%)', -15, 15, 0)
oil_value = st.sidebar.slider('Oil CO2 output (%)', -15, 15, 0)

# Apply percentage changes to baseline values
BASELINE_YEAR = 2022
years = [2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030]
input_rows = []
for country in g7:
    country_baseline = baseline[baseline['country'] == country]
    if not country_baseline.empty:
        for year in years:
            input_rows.append({
                'year': year,
                'country': country,
                'gas': country_baseline['gas'].iloc[0] * (1 + gas_value/100) ** (year - BASELINE_YEAR),
                'coal': country_baseline['coal'].iloc[0] * (1 + coal_value/100) ** (year - BASELINE_YEAR),
                'oil': country_baseline['oil'].iloc[0] * (1 + oil_value/100) ** (year - BASELINE_YEAR)
            })
input_data = pd.DataFrame(input_rows)

# Make prediction using the model's pipeline
# The model is a Pipeline object that includes both preprocessing and Ridge regression
prediction = pd.DataFrame(model.predict(input_data), columns=['temp'])

# Append the prediction to the dataframe
prediction_df = pd.concat([input_data, prediction], axis=1)
df = pd.concat([df, prediction_df], axis=0, ignore_index=True)

# Filter the dataframe based on selected countries
filtered_df = df[df['country'].isin(selected_countries)]

# Plot the data
# Create a selection that chooses the nearest point & selects based on x-value
zoom = alt.selection_interval(
    bind='scales',
    encodings=['x', 'y']
)

historical_line = alt.Chart(filtered_df[filtered_df['year'] <= BASELINE_YEAR]).mark_line().encode(
    x=alt.X("year:O", title="Year"),
    y=alt.Y("temp:Q", title="Temperature Increase (°C)"),
    color=alt.Color("country:N", title="Country")
)

historical_points = alt.Chart(filtered_df[filtered_df['year'] <= BASELINE_YEAR]).mark_circle(size=50).encode(
    x=alt.X("year:O"),
    y=alt.Y("temp:Q"),
    color=alt.Color("country:N"),
    tooltip=[
        alt.Tooltip("country:N", title="Country"),
        alt.Tooltip("year:O", title="Year"),
        alt.Tooltip("temp:Q", title="Temperature Increase (°C)", format=".2f")
    ]
)

# Combine charts based on show_predictions checkbox
if show_predictions:
    predictions_points = alt.Chart(filtered_df[filtered_df['year'] >= BASELINE_YEAR]).mark_circle(size=50).encode(
        x=alt.X("year:O"),
        y=alt.Y("temp:Q"),
        color=alt.Color("country:N"),
        tooltip=[
            alt.Tooltip("country:N", title="Country"),
            alt.Tooltip("year:O", title="Year"),
            alt.Tooltip("temp:Q", title="Temperature Increase (°C)", format=".2f")
        ]
    )
    plot = (historical_line + historical_points + predictions_points)
else:
    plot = (historical_line + historical_points)

plot = plot.properties(
    width=800,
    height=400
).configure_axis(
    titleFontSize=20,
    labelFontSize=16
).add_selection(zoom)

# Main title and plot
st.title('Temperature Increase Prediction from 2023 to 2030')

# Add explanatory text
st.markdown("""
This interactive dashboard predicts temperature increases based on changes in CO2 emissions from different sources (gas, coal, and oil) for G7 countries.

- Temperature predictions are based on a machine learning model trained on historical data
- Temperature increase are compared to pre-industrial levels (pre-1900s)
- Solid lines show historical data up to 2022
- Dotted lines show predicted temperature increases based on your selected CO2 changes
""")

st.altair_chart(plot, use_container_width=True)

# Display predictions table only if show_predictions is True
if show_predictions:
    st.subheader("Predicted Temperature Increases")
    predictions_df = filtered_df[filtered_df['year'] >= 2023].copy()
    # Round numerical columns to 3 decimal places
    predictions_df['temp'] = predictions_df['temp'].round(3)
    predictions_df['gas'] = predictions_df['gas'].round(3)
    predictions_df['coal'] = predictions_df['coal'].round(3)
    predictions_df['oil'] = predictions_df['oil'].round(3)
    # Reorder columns for better readability
    predictions_df = predictions_df[['country', 'year', 'temp', 'gas', 'coal', 'oil']]
    st.dataframe(
        predictions_df,
        column_config={
            "country": "Country",
            "year": "Year",
            "temp": "Temperature Increase (°C)",
            "gas": "Gas CO2 (per capita)",
            "coal": "Coal CO2 (per capita)",
            "oil": "Oil CO2 (per capita)"
        },
        hide_index=True
    )