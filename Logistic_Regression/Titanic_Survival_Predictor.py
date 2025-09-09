import numpy as np
import pandas as pd

# loading data
df = pd.read_csv(r"C:\Users\user\Downloads\train.csv")
print(df.head())


# region PREPROCESSING DATA
df.drop(["PassengerId"], axis= 1, inplace= True)
df.drop(["Name"], axis = 1, inplace= True)
df.drop(["Ticket"], axis = 1, inplace= True)
df.drop(["Cabin"], axis = 1, inplace= True)
df.dropna(inplace= True)
# now for the one hot encoding for the gender
# gender_column = pd.get_dummies(df["Sex"]).astype(int)
# make it binary encoding instead for to make the model less complex
male_column = pd.get_dummies(df["Sex"], drop_first= True, prefix= "Is").astype(int)
df = pd.concat([df, male_column], axis = 1)
# now we can get rid of the sex column
df.drop(["Sex"], axis = 1, inplace= True)
df.drop(["Embarked"], axis = 1, inplace= True)
# now for the correlation to know which ones are redundant 
correlations_table = df.corr()
print(correlations_table.head())
# THIS SHOWED THAT [PCLASS, ISMALE, FARE] ARE THE ONLY ONES CORRELATED WITH SURVIVED
# now check for the corr between them to remove ones that very corr with each other
# well the pclass is only corr with ismale by 0.15 so we can take both for now
# pclass and fare are highly corr with each other by 0.55 we can use only one of them to make things simpler

df.drop(["Age", "SibSp", "Parch", "Fare"], axis = 1, inplace= True)

print(df.head())
# endregion

# region DEFINING FUNCTIONS