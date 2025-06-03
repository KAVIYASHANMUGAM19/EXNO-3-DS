## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation 
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

```
NAME: KAVIYA S
REG NO.: 212223040090
```
```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```



![image](https://github.com/user-attachments/assets/fb4b9c8a-d64f-4216-bbe1-d8cc693e987a)



```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```



![image](https://github.com/user-attachments/assets/4899f92b-b4c8-4778-b2ac-96b675069ab3)



```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```


![image](https://github.com/user-attachments/assets/38193e93-27ad-4cc9-a193-f974cf3fc344)



```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```



![image](https://github.com/user-attachments/assets/bce24d7e-87ea-40e4-937c-d7540bd26b96)



```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```


![image](https://github.com/user-attachments/assets/4c9cbd4f-261b-4b31-8650-449fd367e4e9)


```
pd.get_dummies(df2,columns=["nom_0"])
```


![image](https://github.com/user-attachments/assets/bbda2cc7-0824-401c-b45a-959af69d7c84)



```
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
dfb=pd.concat([df,nd],axis=1)
dfb
```


![image](https://github.com/user-attachments/assets/9f772f77-2ba1-4acc-bea5-af45f84283a4)



```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```


![image](https://github.com/user-attachments/assets/d019b565-0f35-494a-9251-be1a61391d0a)


```
import pandas as pd
from scipy import stats
```


![image](https://github.com/user-attachments/assets/d03bf85d-cb01-43af-80d8-29970d0e7048)


```
df.skew()
```


![image](https://github.com/user-attachments/assets/1e688fa2-aab0-4fc9-b146-8aedfcb7921c)


```
np.log(df["Highly Positive Skew"])
```


![image](https://github.com/user-attachments/assets/46fed48d-de60-49ec-9b71-8bbd4e3cbc00)

```
np.reciprocal(df["Moderate Positive Skew"])
```

![image](https://github.com/user-attachments/assets/10c48c4d-8a83-4336-941b-e02824147264)


```
np.sqrt(df["Highly Positive Skew"])
```

![image](https://github.com/user-attachments/assets/208d79cd-2f82-4192-a49b-2c4c71fe2c01)

``
np.square(df["Highly Positive Skew"])
``

![image](https://github.com/user-attachments/assets/a4b8526d-9805-4e56-836e-3d4387887c1f)

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

![image](https://github.com/user-attachments/assets/7c70f349-6185-473a-a26a-2dab978af2e2)

```
from scipy import stats
df["Highly Negative Skew_yeojohnson"], parameters = stats.yeojohnson(df["Highly Negative Skew"])
print(df.skew())
```

![image](https://github.com/user-attachments/assets/ddd37e7d-eb69-4247-a3a2-77fa3fd7199c)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```

![image](https://github.com/user-attachments/assets/ec7f780d-92b0-41d6-b012-a3faf48489d7)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/b8405778-4ea8-4392-9404-ee379711833a)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/7c091ab1-8145-4942-9814-8cdbaa6f73a0)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/a4a4aa73-fba5-4321-969e-52b3f18ca591)


```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/0d516a0f-518d-46f7-aa13-f06a3b76b545)

```
dt = pd.read_csv("titanic_dataset.csv")
dt = dt.dropna(subset=["Age"])
qt = QuantileTransformer(output_distribution='normal', n_quantiles=dt.shape[0])
dt["Age_1"] = qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'], line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/4878c882-dfc5-4b49-9af3-b9b92fbd02dc)


```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/04972b77-938f-4f6c-a7e4-0607aaabe37b)



# RESULT:
Feature Encoding, Transformation process and saving the data to a file is done successfully

       
