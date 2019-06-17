# An Embarrassment of Pandas

![group-of-pandas](https://www.gannett-cdn.com/-mm-/8ec5d09776cb16d4fc0180df562106e57760eb95/c=0-148-4253-2551/local/-/media/2018/04/03/USATODAY/USATODAY/636583772913864667-XXX-PANDAS-PDS-00508-98906967.JPG?width=3200&height=1680&fit=crop)

* [DataFrames](#dataframes)
* [Series](#series)
* [Method Chaining](#method-chaining)
* [Aggregation](#aggregation)
* [New Columns](#new-columns)
* [Feature Engineering](#feature-engineering)

## DataFrames

* Options
```python
# See more columns
pd.set_option("display.max_columns", 500)

# See more rows
pd.set_option("display.max_rows", 500)

# Round floats to 3 decimal places, in lieu of scientific notation
pd.set_option("display.float_format", lambda x: "%.3f" % x)

# Increase column width
pd.set_option('max_colwidth', 50)
```

* Advanced `read_csv()` options
```python
pd.read_csv(
    'some_data.csv',
    skiprows = 2,
    na_values = [],
    parse_dates = [],
    
)
```

* Reading in multiple files at once
```python
import glob

df = pd.concat([pd.read_csv(f) for f in glob.glob("*.csv")])

# More `read_csv()` options
df = pd.concat([pd.read_csv(f, encoding = 'latin1') for f in glob.glob("*.csv")])
```

* Column headers
```python
```

* Correlation between all
```python
df.corr()
```

* Filtering DataFrame - using `pd.Series.isin()`
```python
df[df['dimension'].isin(['A', 'B', 'C'])]
```

* Filtering DataFrame - using `df.query()`
```python
df.query('A > C')
```

## Series

## Method Chaining

## Aggregation

## New Columns

* Based on one conditions - using `np.where()`
```python
np.where(df['gender'] == 'Male', 1, 0)
```

* Based on multiple conditions - using `np.where()`
```python
np.where(df['measure'] < 5, 'Low', np.where(df['measure'] < 10, 'Medium', 'High'))
```

* Based on multiple conditions - using `pd.cut()`
```python
```

* Based on multiple conditions - using `np.select()`
```python
conditions 
```

* Based on multiple conditions - using `pd.Series.map()`
```python
values = {'Low': 1, 'Medium': 2, 'High': 3}

df['dimension'].map(values)
```

## Feature Engineering

* Date and time
```python
```

* Count occurence of dimension
```python
df.groupby('dimension', as_index = False).transform(len)

# By another column
df.groupby('dimension', as_index = False)['another_dimension'].transform(len)
```

* Count total of measure
```python
df.groupby('dimension', as_index = False)['measure'].sum()
```

* Aggregate statistics for numeric columns only
```python
df.groupby('dimension', as_index = False).agg(['count', 'mean', 'max', 'min', 'sum'])
```

* Binning numerical value
```python
pd.qcut(data['measure'], q = 4, labels = False)
```

* Dummy variables
```python
# Use `drop_first = True` to avoid multicollinearity
pd.get_dummies(df, drop_first = True)
```

* Sort and take first value by some dimension
```python
df.sort_values(by = "variable").groupby("dimension", as_index = False).first()
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
