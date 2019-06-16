# An Embarrassment of Pandas

![group-of-pandas](https://www.gannett-cdn.com/-mm-/8ec5d09776cb16d4fc0180df562106e57760eb95/c=0-148-4253-2551/local/-/media/2018/04/03/USATODAY/USATODAY/636583772913864667-XXX-PANDAS-PDS-00508-98906967.JPG?width=3200&height=1680&fit=crop)

* [DataFrames](#dataframes)
* [Series](#series)
* [Aggregation](#aggregation)
* [New Columns](#new-columns)
* [Feature Engineering](#feature-engineering)

## DataFrames

* Advanced `read_csv()` options
```python
```

* Reading in multiple files at once
```python
import glob

df = pd.concat([pd.read_csv(f) for f in glob.glob("*.csv")])

# More `read_csv()` options
df = pd.concat([pd.read_csv(f, encoding = 'latin1') for f in glob.glob("*.csv")])
```

## Series

## Aggregation

## New Columns

## Feature Engineering

* Value count of column
```python
df.groupby('some_dimension').transform(len)
```

* RFM - Recency, Frequency & Monetary
```python
```

* Haversine
```python
import numpy as np
from numpy import pi, deg2rad, cos, sin, arcsin, sqrt

def haversine(s_lat, s_lng, e_lat, e_lng):
    """
    determines the great-circle distance between two point
    on a sphere given their longitudes and latitudes
    """

    # approximate radius of earth in miles
    R = 3959.87433

    s_lat = s_lat * np.pi / 180.0
    s_lng = np.deg2rad(s_lng)
    e_lat = np.deg2rad(e_lat)
    e_lng = np.deg2rad(e_lng)

    d = (
        np.sin((e_lat - s_lat) / 2) ** 2
        + np.cos(s_lat) * np.cos(e_lat) * np.sin((e_lng - s_lng) / 2) ** 2
    )

    return 2 * R * np.arcsin(np.sqrt(d))

# Convert pd.Series() -> np.ndarray()
df['distance'] = haversine(df['start_lat'].values, df['start_long'].values, df['end_lat'].values, df['end_long'].values)
```
