import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("day.csv")

#x --> Independent
#y --> Dependent
df.columns

from sklearn.linear_model import LinearRegression

lm = LinearRegression()
#lm.fit(df["hum"].values.reshape(-1,1),df["cnt"].values.reshape(-1,1))


#lm.predict(0.8)
#lm.predict(0.6)
#shows inversely proportional relationship

plt.scatter(df["hum"], df["cnt"])

lm.fit(df[["temp", "hum"]],df["cnt"].value.reshape(-1,1))
lm.predict([[0.2, 0.8]])

