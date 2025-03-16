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
years = [2023, 2024, 2025, 2026]
input_rows = []
for country in g7:
    country_baseline = baseline[baseline['country'] == country]
    if not country_baseline.empty:
        for year in years:
            input_rows.append({
                'year': year,
                'country': country,
                'gas': country_baseline['gas'].iloc[0] * (1 + gas_value/100) * (year - BASELINE_YEAR),
                'coal': country_baseline['coal'].iloc[0] * (1 + coal_value/100) * (year - BASELINE_YEAR),
                'oil': country_baseline['oil'].iloc[0] * (1 + oil_value/100) * (year - BASELINE_YEAR)
            })
input_data = pd.DataFrame(input_rows)

# Make prediction using the Ridge model
prediction = pd.DataFrame(model.predict(input_data), columns=['temp'])

# Append the prediction to the dataframe
prediction_df = pd.concat([input_data, prediction], axis=1)
df = pd.concat([df, prediction_df], axis=0, ignore_index=True)

# Filter the dataframe based on selected countries
filtered_df = df[df['country'].isin(selected_countries)]
filtered_df["temp_increase"] = filtered_df.groupby("country")["temp"].cumsum()

# Plot the data
historical = alt.Chart(filtered_df[filtered_df['year'] <= BASELINE_YEAR]).mark_line().encode(
    x=alt.X("year:O", title="Year"),
    y=alt.Y("temp_increase:Q", title="Temperature Increase (°C)"),
    color=alt.Color("country:N", title="Country")
)

predictions = alt.Chart(filtered_df[filtered_df['year'] >= BASELINE_YEAR]).mark_line(strokeDash=[4, 4]).encode(
    x=alt.X("year:O"),
    y=alt.Y("temp_increase:Q"),
    color=alt.Color("country:N")
)

plot = (historical + predictions).properties(
    width=800,
    height=400
).configure_axis(
    titleFontSize=20,
    labelFontSize=16
)

# Main title and plot
st.title('Temperature Increase Prediction from 2023 to 2028')

# Add explanatory text
st.markdown("""
This interactive dashboard predicts temperature increases based on changes in CO2 emissions from different sources (gas, coal, and oil) for G7 countries.

- Temperature predictions are based on a machine learning model trained on historical data
- Temperature increase are compared to pre-industrial levels (pre-1900s)
- Solid lines show historical data up to 2022
- Dotted lines show predicted temperature increases based on your selected CO2 changes
""")

st.altair_chart(plot, use_container_width=True)