# GeoAI — world air quality

- 3D satellite globe with ~140 cities (live + 3-day forecast); click anywhere for that spot's values
- Selection bar: data source, city (or any place), pollutant/product, calendar time frame → map + report
- Data sources: CAMS model (no setup), Sentinel-5P TROPOMI and MODIS MAIAC via Google Earth Engine
- QGIS export: ready-to-open package (grid + style, city layer, CSV)
- Assistant: averages, trends, comparisons, weather links and machine learning — just type

Run (from this folder):
    python -m pip install -r requirements.txt
    python -m streamlit run app.py

For satellite sources (one time):  earthengine authenticate   then paste your Earth Engine project ID in the app.
Optional: python -m pip install xgboost
