import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

df = pd.read_csv(r'C:\Users\eleni\PycharmProjects\Seismos-Santorini\Data\catalogue.csv')
print(df.columns)

df['Time'] = pd.to_datetime(df['Time'])

fig = px.scatter_mapbox(df,
                        lat="Latitude",
                        lon="Longtitude",
                        color="Magnitude",
                        size="Magnitude",
                        hover_name="Time",
                        hover_data={"Depth": True, "Magnitude": True},
                        color_continuous_scale=px.colors.sequential.Inferno,
                        zoom=9, # Adjust zoom level for Santorini area
                        mapbox_style="carto-positron",
                        title="Earthquake Locations around Santorini by Magnitude and Depth")
fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
fig.show()
fig.write_html(r"C:\Users\eleni\PycharmProjects\Seismos-Santorini\Data\Plots\map.html")