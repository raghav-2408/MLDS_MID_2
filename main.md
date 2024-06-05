# week 8

```python
import pandas as pd
d = pd.read_csv("train.csv")
print(d.head(3))
```

```python
shape_data = d.shape
print(shape_data) # rows and columns
```

```python
print(d['Loan_Status'])
```

```python
# correlation - 
x = d.corr
print(x)
```

```python
independent_variables = d.drop(['Married'], axis = 1)
print(independent_variables)
# assume Married to be as a dependent and we just remove it
```

# week 9

```python
# print important parameters
import pandas as pd
import numpy as np

d = pd.read_csv('water_dataX.csv', encoding = 'latin1')

print(d.columns)

important_para = ['STATION CODE', 'LOCATIONS', 'STATE']
res = d[important_para]
print(res)
```


```python
# print the conversions
d.dtypes
```
```python
# Calculate pH, BDO, TO

pH = d['PH'].astype('float')
res = pH.mean()
print(res)

```


```python
# print the water quality index

start = 1
end = 1779
ph = d.iloc[start:end, 5]
bod = d.iloc[start:end, 7].astype(np.float64)
do = d.iloc[start:end, 4].astype(np.float64)


d1 = pd.concat([do, ph, bod], axis = 1)
d1.columns = ['do','ph', 'bod']


print(d1)

# this is not correct ...
```


# Week 10


```python
import pandas as pd
d = pd.read_csv('matches.csv')
res = d['winner'].value_counts()
print(res)
```
```python
lenn = len(res)
print(lenn)
```
```python
for i in d.columns:
  temp = d[i].unique()
  print(f"Unique values : {temp}")
```
```python
null_values = d.isnull().sum()
print(null_values)
```
```python
# Random Forest tends to generalize well to unseen data and is less prone to overfitting, which is beneficial when dealing with real-world datasets like IPL matches.
```
