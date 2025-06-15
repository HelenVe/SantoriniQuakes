import pandas as pd
import plotly.express as px

df = pd.read_csv(r'C:\Users\eleni\PycharmProjects\Seismos-Santorini\Data\catalogue.csv')


df['Time'] = pd.to_datetime(df['Time'])
print("Earliest date", df['Time'].min())
print("Latest date", df['Time'].max())

# pick 2025 dates

df = df[df['Time'].dt.date > pd.to_datetime('2025-01-01').date()]
# -------------------- Histogram of Magnitude
fig = px.histogram(df, x="Magnitude", nbins=20, title="Distribution of Earthquake Magnitudes",
                   labels={"Magnitude": "Earthquake Magnitude"}, color_discrete_sequence=['indianred'])
fig.update_layout(bargap=0.1)
fig.show()
fig.write_html(r'C:\Users\eleni\PycharmProjects\Seismos-Santorini\Data\Plots\dstr_magnitudes.html')

# -------------  Magnitude over Time
fig = px.line(df.sort_values(by='Time'),x="Time", y="Magnitude", title="Earthquake Magnitude Over Time",
              labels={"Time": "Date and Time", "Magnitude": "Earthquake Magnitude"})
fig.update_traces(mode='lines+markers', marker=dict(size=5))
fig.show()
fig.write_html(r'C:\Users\eleni\PycharmProjects\Seismos-Santorini\Data\Plots\magnitudes_time.html')

# Mean Daily magnitutde

df_daily = df.set_index('Time')
daily_mean_magnitude = df_daily['Magnitude'].resample('D').mean().reset_index()
daily_mean_magnitude.columns = ['Date', 'Mean Magnitude']
fig = px.line(daily_mean_magnitude, x="Date", y="Mean Magnitude",
              title="Daily Mean Earthquake Magnitude Over Time", labels={"Date": "Date", "Mean Magnitude": "Mean Earthquake Magnitude"},
              line_shape='linear'
             )

fig.update_traces(mode='lines+markers', marker=dict(size=4))
fig.update_layout(xaxis_title="Date", yaxis_title="Mean Magnitude", hovermode="x unified")
fig.show()
fig.write_html(r'C:\Users\eleni\PycharmProjects\Seismos-Santorini\Data\Plots\daily_mean_magnitudes_time.html')

# --------------- Magnitude vs. Depth
fig = px.scatter(df,x="Magnitude",y="Depth",color="Time", size="Magnitude",hover_name="Time",
                 title="Magnitude vs. Depth of Earthquakes",
                 labels={"Magnitude": "Earthquake Magnitude", "Depth": "Earthquake Depth (km)"})
fig.show()
fig.write_html(r'C:\Users\eleni\PycharmProjects\Seismos-Santorini\Data\Plots\magnitudes_depth.html')
