
# Market Basket Analysis of Store Data

## Dataset Description

* Data of 3 Million customer orders of Instacart was taken from Kaggle.
* We have library(**apyori**) to calculate the association rule using Apriori.

## Import the Library


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from apyori import apriori
```

## Read data and Display


```python
store_data = pd.read_csv("store_data.csv", header=None)
display(store_data.head())
print(store_data.shape)
```


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>shrimp</td>
      <td>almonds</td>
      <td>avocado</td>
      <td>vegetables mix</td>
      <td>green grapes</td>
      <td>whole weat flour</td>
      <td>yams</td>
      <td>cottage cheese</td>
      <td>energy drink</td>
      <td>tomato juice</td>
      <td>low fat yogurt</td>
      <td>green tea</td>
      <td>honey</td>
      <td>salad</td>
      <td>mineral water</td>
      <td>salmon</td>
      <td>antioxydant juice</td>
      <td>frozen smoothie</td>
      <td>spinach</td>
      <td>olive oil</td>
    </tr>
    <tr>
      <th>1</th>
      <td>burgers</td>
      <td>meatballs</td>
      <td>eggs</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>chutney</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>turkey</td>
      <td>avocado</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mineral water</td>
      <td>milk</td>
      <td>energy bar</td>
      <td>whole wheat rice</td>
      <td>green tea</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


    (7501, 20)
    

## Preprocessing on Data
*  Here we need a data in form of list for Apriori Algorithm.


```python
records = []
for i in range(1, 7501):
    records.append([str(store_data.values[i, j]) for j in range(0, 20)])
```


```python
print(type(records))
```

    <class 'list'>
    


```python
association_rules = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)
association_results = list(association_rules)
```

## How many relation derived


```python
print("There are {} Relation derived.".format(len(association_results)))
```

    There are 48 Relation derived.
    

### Association Rules Derived


```python
for i in range(0, len(association_results)):
    print(association_results[i][0])
```

    frozenset({'light cream', 'chicken'})
    frozenset({'escalope', 'mushroom cream sauce'})
    frozenset({'escalope', 'pasta'})
    frozenset({'herb & pepper', 'ground beef'})
    frozenset({'tomato sauce', 'ground beef'})
    frozenset({'olive oil', 'whole wheat pasta'})
    frozenset({'shrimp', 'pasta'})
    frozenset({'nan', 'light cream', 'chicken'})
    frozenset({'shrimp', 'chocolate', 'frozen vegetables'})
    frozenset({'cooking oil', 'spaghetti', 'ground beef'})
    frozenset({'escalope', 'mushroom cream sauce', 'nan'})
    frozenset({'escalope', 'pasta', 'nan'})
    frozenset({'spaghetti', 'ground beef', 'frozen vegetables'})
    frozenset({'milk', 'olive oil', 'frozen vegetables'})
    frozenset({'shrimp', 'mineral water', 'frozen vegetables'})
    frozenset({'spaghetti', 'olive oil', 'frozen vegetables'})
    frozenset({'shrimp', 'spaghetti', 'frozen vegetables'})
    frozenset({'spaghetti', 'frozen vegetables', 'tomatoes'})
    frozenset({'spaghetti', 'ground beef', 'grated cheese'})
    frozenset({'herb & pepper', 'ground beef', 'mineral water'})
    frozenset({'herb & pepper', 'nan', 'ground beef'})
    frozenset({'herb & pepper', 'spaghetti', 'ground beef'})
    frozenset({'milk', 'ground beef', 'olive oil'})
    frozenset({'nan', 'tomato sauce', 'ground beef'})
    frozenset({'shrimp', 'spaghetti', 'ground beef'})
    frozenset({'milk', 'spaghetti', 'olive oil'})
    frozenset({'soup', 'mineral water', 'olive oil'})
    frozenset({'nan', 'olive oil', 'whole wheat pasta'})
    frozenset({'shrimp', 'nan', 'pasta'})
    frozenset({'spaghetti', 'pancakes', 'olive oil'})
    frozenset({'shrimp', 'chocolate', 'frozen vegetables', 'nan'})
    frozenset({'cooking oil', 'nan', 'spaghetti', 'ground beef'})
    frozenset({'nan', 'spaghetti', 'ground beef', 'frozen vegetables'})
    frozenset({'milk', 'spaghetti', 'mineral water', 'frozen vegetables'})
    frozenset({'milk', 'nan', 'olive oil', 'frozen vegetables'})
    frozenset({'shrimp', 'nan', 'mineral water', 'frozen vegetables'})
    frozenset({'nan', 'spaghetti', 'olive oil', 'frozen vegetables'})
    frozenset({'shrimp', 'nan', 'spaghetti', 'frozen vegetables'})
    frozenset({'nan', 'spaghetti', 'frozen vegetables', 'tomatoes'})
    frozenset({'nan', 'spaghetti', 'ground beef', 'grated cheese'})
    frozenset({'herb & pepper', 'nan', 'ground beef', 'mineral water'})
    frozenset({'herb & pepper', 'nan', 'spaghetti', 'ground beef'})
    frozenset({'milk', 'nan', 'ground beef', 'olive oil'})
    frozenset({'shrimp', 'nan', 'spaghetti', 'ground beef'})
    frozenset({'milk', 'nan', 'spaghetti', 'olive oil'})
    frozenset({'nan', 'soup', 'mineral water', 'olive oil'})
    frozenset({'nan', 'spaghetti', 'pancakes', 'olive oil'})
    frozenset({'milk', 'frozen vegetables', 'nan', 'spaghetti', 'mineral water'})
    

## Rules Generated


```python
for item in association_results:
    # first index of the inner list
    # Contains base item and add item
    pair = item[0]
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    # second index of the inner list
    print("Support: " + str(item[1]))

    # third index of the list located at 0th
    # of the third index of the inner list


    
