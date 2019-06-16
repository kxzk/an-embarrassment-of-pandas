# An Embarrassment of Pandas

![group-of-pandas](https://www.gannett-cdn.com/-mm-/8ec5d09776cb16d4fc0180df562106e57760eb95/c=0-148-4253-2551/local/-/media/2018/04/03/USATODAY/USATODAY/636583772913864667-XXX-PANDAS-PDS-00508-98906967.JPG?width=3200&height=1680&fit=crop)

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

## Cleaning

## Feature Engineering

* RFM - Recency, Frequency & Monetary
```python
```

* Haversine
```python
```

## Aggregation
