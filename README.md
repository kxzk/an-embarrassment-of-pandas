# An Embarrassment of Pandas

![group-of-pandas](https://www.gannett-cdn.com/-mm-/8ec5d09776cb16d4fc0180df562106e57760eb95/c=0-148-4253-2551/local/-/media/2018/04/03/USATODAY/USATODAY/636583772913864667-XXX-PANDAS-PDS-00508-98906967.JPG?width=3200&height=1680&fit=crop)

Why an embarrassment? Because it's the name for a [group of pandas!](https://www.reference.com/pets-animals/group-pandas-called-71cd65ea758ca2e2)

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

* Useful `read_csv()` options
```python
pd.read_csv(
    'some_data.csv',
    skiprows = 2,
    usecols = [],
    dtype = [],
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
# Lower all values
df.columns = [x.lower() for x in df.columns]
```

* Correlation between all
```python
df.corr()
```

* Filtering DataFrame - using `pd.Series.isin()`
```python
df[df['dimension'].isin(['A', 'B', 'C'])]
```

* Filtering DataFrame - using `pd.Series.str.contains()`
```python
df[df['dimension'].str.contains('word')]
```

* Filtering DataFrame - using `df.query()`
```python
df.query('A > C')
```

* Joining
```python
```

* Styling with dollar signs, commas and percent signs
```python
styling_options = {'sales': '${0:,.0f}', 'percent_of_sales': '{:.2%f}'}

df.style.format(styling_options)
```

* Add highlighting for max and min values
```python
df.style.highlight_max(color = 'lightgreen').highlight_min(color = 'red')
```

* Conditional formatting for one column
```python
df.style.background(subset = ['measure'], cmap = 'viridis')
```

## Series

* Value counts as percentages
```python
df['measure'].value_counts(normalize = True)
```

## Method Chaining

```python
```

[Recommended Read - Effective Pandas](https://leanpub.com/effective-pandas)

## Aggregation

* By month
```python
df.groupby([pd.Grouper(key = 'date', freq = 'M')])['measure'].agg(['sum', 'mean'])
```

## New Columns

* Based on one condition - using `np.where()`
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

* Based on manual mapping - using `pd.Series.map()`
```python
values = {'Low': 1, 'Medium': 2, 'High': 3}

df['dimension'].map(values)
```

* Automatically create dictionary from dimension values
```python
dimension_mappings = {v: k for k, v in enumerate(dimension.unique())}

df['dimension'].map(dimension_mappings)
```

* Using list comprehensions
```python
# Grabbing domain name from email
df['domain'] = [x.split('@')[1] for x in df['email']]
```

## Feature Engineering

* Date and time
```python
```

* Count occurence of dimension
```python
df.groupby('dimension').transform(len)

# By another column
df.groupby('dimension')['another_dimension'].transform(len)
```

* Count total of measure
```python
df.groupby('dimension')['measure'].sum()
```

* Distinct list aggregation
```python
df[['customer_id', 'products']].drop_duplicates().groupby('customer_id')['products'].apply(list)
```

* Aggregate statistics for numeric columns only
```python
df.groupby('dimension').agg(['count', 'mean', 'max', 'min', 'sum'])
```

* Binning numerical value
```python
pd.qcut(data['measure'], q = 4, labels = False)
```

* Dummy variables
```python
# Use `drop_first = True` to avoid collinearity
pd.get_dummies(df, drop_first = True)
```

* Sort and take first value by some dimension
```python
df.sort_values(by = "variable").groupby("dimension").first()
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
