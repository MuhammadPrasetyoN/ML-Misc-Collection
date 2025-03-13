# 📌 Encoding in Machine Learning

## 📖 What is Encoding in Machine Learning?
Encoding is the process of converting categorical data into numerical format so that machine learning algorithms can process it efficiently. Many machine learning models work better with numerical data rather than categorical data.

---

## 🔹 Types of Encoding in Machine Learning

### 1️⃣ Label Encoding
Label Encoding assigns a unique integer to each category. It is useful for ordinal data (ordered categories), but it can introduce an unintended ordinal relationship for nominal data.

#### ✅ Example:
| City  | Encoded |
|--------|---------|
| Paris  | 0       |
| London | 1       |
| Berlin | 2       |
| Paris  | 0       |

#### 💻 Code:
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cities = ['Paris', 'London', 'Berlin', 'Paris']
encoded_cities = le.fit_transform(cities)
print(encoded_cities)  # Output: [0 1 2 0]
```

---

### 2️⃣ One-Hot Encoding (OHE)
One-Hot Encoding creates a binary column for each category. This prevents the model from misinterpreting ordinal relationships.

#### ✅ Example:
| City   | Paris | London | Berlin |
|--------|-------|--------|--------|
| Paris  | 1     | 0      | 0      |
| London | 0     | 1      | 0      |
| Berlin | 0     | 0      | 1      |

#### 💻 Code:
```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np

ohe = OneHotEncoder(sparse_output=False)
cities = np.array(['Paris', 'London', 'Berlin', 'Paris']).reshape(-1, 1)
encoded_cities = ohe.fit_transform(cities)
print(encoded_cities)
```

---

### 3️⃣ Ordinal Encoding
Similar to Label Encoding but specifically used for data with a clear order (e.g., education level: High School < Bachelor < Master < PhD).

#### ✅ Example:
| Education Level | Encoded |
|----------------|---------|
| High School    | 0       |
| Bachelor       | 1       |
| Master        | 2       |
| PhD           | 3       |

#### 💻 Code:
```python
from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder(categories=[['High School', 'Bachelor', 'Master', 'PhD']])
education = np.array(['Master', 'PhD', 'High School', 'Bachelor']).reshape(-1, 1)
encoded_education = oe.fit_transform(education)
print(encoded_education)
```

---

### 4️⃣ Frequency Encoding
Replaces each category with its occurrence count in the dataset. Useful when dealing with a large number of categories.

#### ✅ Example:
| Country | Frequency |
|---------|----------|
| USA     | 3        |
| Canada  | 2        |
| UK      | 1        |
| USA     | 3        |

#### 💻 Code:
```python
import pandas as pd

data = pd.DataFrame({'Country': ['USA', 'Canada', 'UK', 'USA', 'Canada', 'USA']})
frequency_encoding = data['Country'].value_counts().to_dict()
data['Country_Encoded'] = data['Country'].map(frequency_encoding)
print(data)
```

---

### 5️⃣ Target Encoding (Mean Encoding)
Each category is replaced by the mean of the target variable within that category. Used in regression or classification tasks.

#### ✅ Example:
| Job  | Target | Encoded |
|------|--------|---------|
| A    | 1      | 0.75    |
| A    | 1      | 0.75    |
| B    | 0      | 0.33    |
| B    | 1      | 0.33    |
| C    | 0      | 0.50    |

#### 💻 Code:
```python
data = pd.DataFrame({'Job': ['A', 'A', 'B', 'B', 'C'], 'Target': [1, 1, 0, 1, 0]})
target_mean_encoding = data.groupby('Job')['Target'].mean().to_dict()
data['Job_Encoded'] = data['Job'].map(target_mean_encoding)
print(data)
```

---

## 🎯 Conclusion
Choosing the right encoding technique is crucial for improving model performance. Use:
-  **Label Encoding** for ordinal data.
-  **One-Hot Encoding** for nominal categorical data.
-  **Ordinal Encoding** for data with a defined order.
-  **Frequency Encoding** for handling high-cardinality categorical features.
-  **Target Encoding** for supervised tasks where categories have a meaningful relationship with the target variable.