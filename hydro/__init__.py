"""BC hydrometric data pipeline.

Ingests water level and discharge data for British Columbia rivers from the
ECCC GeoMet OGC API (api.weather.gc.ca), stores it in SQLite, computes rolling
statistics with z-score anomaly flags and confidence intervals, and renders a
self-contained HTML report (Plotly time series + folium station map).
"""

__version__ = "0.1.0"
