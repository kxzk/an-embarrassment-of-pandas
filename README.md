# An Embarrassment of Pandas

![group-of-pandas](https://i.imgur.com/BJ1Zss2.png)

Why an embarrassment? Because it's the name for a [group of pandas!](https://www.reference.com/pets-animals/group-pandas-called-71cd65ea758ca2e2)

* [DataFrames](#dataframes)
* [Series](#series)
* [Missing Values](#missing-values)
* [Method Chaining](#method-chaining)
* [Aggregation](#aggregation)
* [New Columns](#new-columns)
* [Feature Engineering](#feature-engineering)

## DataFrames

* Options - [documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/options.html)
```python
# See more columns
pd.set_option("display.max_columns", 500)

# See more rows
pd.set_option("display.max_rows", 500)

# Floating point output precision
pd.set_option("display.precision", 3)

# Increase column width
pd.set_option('max_colwidth', 50)
```

* Useful `read_csv()` options - [documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)
```python
pd.read_csv(
    'data.csv.gz',
    delimiter = "^",
    # line numbers to skip (i.e. headers in an excel report)
    skiprows = 2,
    # used to denote the start and end of a quoted item
    quotechar = "|",
    # return a subset of columns
    usecols = ["return_date", "company", "sales"],
    # data type for data or columns
    dtype = { "sales": np.float64 },
    # additional strings to recognize as NA/NaN
    na_values = [".", "?"],
    # convert to datetime, instead of object
    parse_dates = ["return_date"],
    # for on-the-fly decompression of on-disk data
    # options - gzip, bz2, zip, xz
    compression = "gzip",
    # encoding to use for reading
    encoding = "latin1",
    # read in a subset of data
    nrows = 100
)
```

* Reading in multiple files at once - [glob documentation](https://docs.python.org/3/library/glob.html)
```python
import glob

# ignore_index = True to avoid duplicate index values
df = pd.concat([pd.read_csv(f) for f in glob.glob("*.csv")], ignore_index = True)

# More `read_csv()` options
df = pd.concat([pd.read_csv(f, encoding = "latin1") for f in glob.glob("*.csv")])
```

* Recursively grab all files in a directory
```python
import os
import glob

files = [os.path.join(root, file)
        for root, dir, files in os.walk("./directory")
        for file in glob.glob("*.csv")]
```

* Reading in data from SQLite3 database
```python
import sqlite3

conn = sqlite3.connect("flights.db")
df = pd.read_sql_query("select * from airlines", conn)
conn.close()
```

* Reading in data from Postgres - [BigQuery](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_gbq.html#pandas.read_gbq), [Snowflake](https://docs.snowflake.net/manuals/user-guide/sqlalchemy.html#snowflake-connector-for-python)
```python
from sqlalchemy import create_engine

# 5439 for Redshift
engine = create_engine("postgresql://user@localhost:5432/mydb")
df = pd.read_sql_query("select * from airlines", engine)
```

* Column headers
```python
# Lower all values
df.columns = [x.lower() for x in df.columns]

# Strip out punctuation, replace spaces and lower
df.columns = df.columns.str.replace("[^\w\s]", "").str.replace(" ", "_").str.lower()

# Condense multiindex columns
df.columns = ["_".join(col).lower() for col in df.columns]
```

* Filtering DataFrame - using `pd.Series.isin()`
```python
df[df["dimension"].isin(["A", "B", "C"])]

# not in
df[~df["dimension"].isin(["A", "B", "C"])]
```

* Filtering DataFrame - using `pd.Series.str.contains()`
```python
df[df["dimension"].str.contains("word")]

# not in
df[~df["dimension"].str.contains("word")]
```

* Filtering DataFrame - using `df.query()` - [documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html)
```python
df.query("salary > 100000")

df.query("name == 'john'")

df.query("name == 'john' | name == 'jack'")

df.query("name == 'john' & salary > 100000")

# Grab top 1% of earners
df.query("salary > salary.quantile(.99)")

# Make more than the mean
df.query("salary > salary.mean()")

# Subset by top 3 most frequent products purchased
df.query("item in item.value_counts().nlargest(3).index")

# Query for null values
df.query("column.isnull()", engine = "python")

# @ - allows you to refer to variables in the environment
names = ["john", "fred", "jack"]
df.query("name in @names")
```

* Joining
```python
# Inner join
pd.merge(df1, df2, on = "key")

# Left join on different key names
pd.merge(df1, df2, right_on = ["right_key"], left_on = ["left_key"], how = "left")
```

* Select columns based on data type
```python
df.select_dtypes(include = "number")
df.select_dtypes(exclude = "number")

df.select_dtypes(include = ["object", "datetime"])
```

* Reverse column order
```python
df.loc[:, ::-1]
```

* Descriptive statistics
```python
# Measures
df.describe(include=[np.number]).T

# Dimensions
df.describe(include=[pd.Categorical]).T

# Add percent frequency for top dimension
df["freq_total"] = df["freq"].div(df["count"])
```

* Styling with dollar signs, commas and percent signs
```python
styling_options = {"sales": "${0:,.0f}", "percent_of_sales": "{:.2%f}"}

df.style.format(styling_options)
```

* Add highlighting for max and min values
```python
df.style.highlight_max(color = "lightgreen").highlight_min(color = "red")
```

* Conditional formatting for one column
```python
df.style.background(subset = ["measure"], cmap = "viridis")
```

## Series

* Value counts as percentages
```python
# Use `dropna = False` to see NaN values
df["meaure"].value_counts(normalize = True, dropna = False)
```

* Replacing errant characters
```python
df["sales"].str.replace("$", "")
```

## Missing Values

* Dropping columns
```python
df.drop(["column_a", "column_b"], axis = 1)
```

* Dropping columns based on NaN threshold - [documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html)
```python
# Any column with 90% missing values will be dropped
df.dropna(thresh = len(df) * .9, axis = 1)
```

* Replacing using `fillna()`
```python
# Impute DataFrame with all zeroes
df.fillna(0)

# Impute column with all zeroes
df["measure"].fillna(0)

# Impute measure with mean of column
df["measure"].fillna(df["measure"].mean())

# Impute dimension with mode of column
df["dimension"].fillna(df["dimension"].mode())

# Impute using a dimension's mean
df["age"].fillna(df.groupby("sex")["age"].transform("mean"))
```

* Replace null value with another column / value, else original column
```python
np.where(pd.isnull(df["dimension"]), df["another_dimension"], df["dimension"])
```

* Replace errant characters with NaN
```python
df.replace(".", np.nan)

# Can also convert 0s for easier cleaning
df.replace(0, np.nan)
```

* Replace numeric values containing a letter with null
```python
df["zipcode"].replace(".*[a-zA-Z].*", np.nan, regex=True)
```

* Drop rows where any value is 0
```python
df[(df != 0).all(1)]
```

* Drop rows where all values are 0
```python
df = df[(df.T != 0).any()]
```

## Method Chaining

```python
(pd.read_csv('employee_salaries.csv')
    .query("salary > 0")
    .assign(sex = lambda df: df["sex"].replace({"female": 1, "male: 0}),
            age = lambda df: pd.cut(df["age"].fillna(df["age"].median()),
                                    bins = [df["age"].min(), 18, 40, df["age"].max()],
                                    labels = ["underage", "young", "experienced"]))
    .rename({"name_1": "first_name", "name_2": "last_name"})
)
```

[Recommended Read - Effective Pandas](https://leanpub.com/effective-pandas)

## Aggregation

* Use `as_index = False` to avoid having to use `reset_index()` everytime
```python
# this
df.groupby("dimension", as_index = False)["measure"].sum()

# not this
df.groupby("dimension")["measure"].sum().reset_index()
```

* By date offset - [full list of options](https://i.imgur.com/KHtdbpc.png)
```python
# H for hours
# D for days
# W for weeks
# WOM for week of month
# Q for quarter end
# A for year end
df.groupby(pd.Grouper(key = "date", freq = "M"))["measure"].agg(["sum", "mean"])
```

* Measure by dimension
```python
# count - number of non-null observations
# sum - sum of values
# mean - mean of values
# mad - mean absolute deviation
# median - arithmetic median of values
# min - minimum
# max - maxmimum
# mode - mode
# std - unbiased standard deviation
# first - first value
# last - last value
df.groupby("dimension")["measure"].sum()

# Specific aggregations by column
df.groupby("dimension").agg({"sales": ["mean", "sum"], "sale_date": "first", "customer": "nunique"})
```

* Aggregate statistics for numeric columns across dimension values
```python
df.groupby("dimension").agg(['count', 'mean', 'max', 'min', 'sum'])
```

## New Columns

* Based on one condition - using `np.where()`
```python
np.where(df["gender"] == "Male", 1, 0)
```

* Based on multiple conditions - using `np.where()`
```python
np.where(df["measure"] < 5, "Low", np.where(df["measure"] < 10, "Medium", "High"))
```

* Based on multiple conditions - using `np.select()`
```python
conditions 
```

* Based on manual mapping - using `pd.Series.map()`
```python
values = {"Low": 1, "Medium": 2, "High": 3}

df["dimension"].map(values)
```

* Automatically create dictionary from dimension values
```python
dimension_mappings = {v: k for k, v in enumerate(df["dimension"].unique())}

df["dimension"].map(dimension_mappings)
```

* Using list comprehensions
```python
# Grabbing domain name from email
df["domain"] = [x.split("@")[1] for x in df["email"]]
```

* Splitting a string column
```python
df["email"].str.split("@", expand = True)[0]
```

* Widening a column - [visual example](https://pandas.pydata.org/pandas-docs/stable/user_guide/reshaping.html)
```python
df.pivot(index = "date", columns = "companies", values = "sales")
```

## Feature Engineering

* Stop using `inplace = True`, it's getting [deprecated](https://github.com/pandas-dev/pandas/issues/16529)

* Screw split-apply-combine, `transform()` it
```python
# this
df["mean_company_salary"] = df.groupby("company")["salary"].transform("mean")

# not this
mean_salary = df.groupby("company")["salary"].agg("mean").rename("mean_salary").reset_index()
df_new = df.merge(mean_salary)
``

* Extracting various date components - [all options](https://i.imgur.com/if2Qosk.png)
```python
df["date"].dt.year
df["date"].dt.quarter
df["date"].dt.month
df["date"].dt.week
df["date"].dt.day
df["date"].dt.weekday
df["date"].dt.weekday_name
df["date"].dt.hour
```

* Time between two dates
```python
# Days between
df["first_date"].sub(df["second_date"]).div(np.timedelta64(1, "D"))

# Months between
df["first_date"].sub(df["second_date"]).div(np.timedelta64(1, "M"))

# Equivalent to above
(df["first_date] - df["second_date"]) / np.timedelta64(1, "M")
```

* Distinct list aggregation
```python
df.groupby("customer_id").agg({"products": "unique"})
```

* Binning numerical value
```python
pd.qcut(data["measure"], q = 4, labels = False)

pd.cut(df["measure"], bins = 4, labels = False)
```

* Dummy variables
```python
# Use `drop_first = True` to avoid multicollinearity
pd.get_dummies(df, drop_first = True)
```

* Sort and take first value by some dimension
```python
df.sort_values(by = "variable").groupby("dimension").first()
```

* Log transformation
```python
# For positive data with no zeroes
np.log(df["sales"])

# For positive data with zeroes
np.log1p(df["sales"])
```

* Boxcox transformation
```python
from scipy import stats

# Must be positive
stats.boxcox(df["sales"])[0]
```

* Z-scores
```python
from scipy import stats
import numpy as np

z = np.abs(stats.zscores(df))
df = df[(z < 3).all(axis = 1)]
```

* Interquartile range (IQR)
```python
q1 = df["salary"].quantile(0.25)
q3 = df["salary"].quantile(0.75)
iqr = q3 - q1

df.query("(@q1 - 1.5 * @iqr) <= salary <= (@q3 + 1.5 * @iqr)")
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
df['distance'] = haversine(df["start_lat"].values, df["start_long"].values, df["end_lat"].values, df["end_long"].values)
```
