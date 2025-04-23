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
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/7a089718-be55-4aef-9d0d-70ebe4101cfb)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/76fa2d0b-70b0-4df6-9837-e9d568fb23a5)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/bd48694f-8aac-4022-88af-2f04a1111cbe)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/6f28d575-0095-485b-a626-97a879fc0111)
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df=df.copy()

enc=pd.DataFrame(ohe.fit_transform(df[['nom_0']]))
enc
```
![image](https://github.com/user-attachments/assets/5e17ca99-a5a9-4b34-9dfa-5849b5742d50)
```
df2=pd.concat([df,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/8c06bbc3-9bda-4ed8-8bbd-3ee254cac341)
```
pip install --upgrade category_encoders
```
![image](https://github.com/user-attachments/assets/0a4d6826-77ee-4cae-9dae-953d7ec7f806)
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
![image](https://github.com/user-attachments/assets/a3b39bd6-a6ab-496e-aa26-6c82b66f96fc)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/user-attachments/assets/55497114-b6e6-4076-a913-cbe6d75e908b)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/51216809-a9e3-4f23-862b-746afa6a9ed7)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/f2b4129b-ad26-4fe3-ac4c-4914ecc8efb7)
```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/0fea850b-6f3c-44d5-8eec-e50b459749c1)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/f7313b9d-007e-465b-8cb9-df1d36b10c07)
```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/b8bf34d7-bbec-427c-9c69-f2841e1d2bb8)
```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/e31a7ee6-6c58-44c8-9126-4e0b7f6a1bbc)
```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/d80b34f8-3180-4ebe-a09f-a5a500198ba6)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/d296149d-104a-459f-be35-58a615c7c859)
```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/8f06be78-0daf-4537-904d-620aab27d83c)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/93d1ac7a-8491-4f88-8ef7-7b187484ace3)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/86288d85-2137-4400-bedf-ca353b45960c)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/3fe921f7-22ea-4ca7-b8a4-10d676121bda)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/400b7946-344f-4b96-a672-9803b5a187e3)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/78d99509-e8c7-47c0-bb2e-d025c05bd337)
```
dt=pd.read_csv("titanic_dataset.csv")
dt
```
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
```
![image](https://github.com/user-attachments/assets/de6fe440-f014-4fa7-b3e6-8c05901172f3)
```

sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/a32e4ce0-8196-47eb-b881-55a79468434b)
# RESULT:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file is obtained successfully

       
