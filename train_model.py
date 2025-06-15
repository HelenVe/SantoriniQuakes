import pandas as pd
import plotly.express as px

df = pd.read_csv(r'C:\Users\eleni\PycharmProjects\Seismos-Santorini\Data\catalogue.csv')


df['Time'] = pd.to_datetime(df['Time'])

df_resampled = df.set_index('Time')
mean_magnitudes_15min = df_resampled['Magnitude'].resample('15min').mean().ffil() # forward fill
print(mean_magnitudes_15min.head())
mean_magnitudes_filled = mean_magnitudes_15min.rolling(window=4, min_periods=1).mean().fillna(0) # take last 4 measurements
