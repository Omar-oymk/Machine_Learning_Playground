import numpy as np
import pandas as pd
# loading data
df = pd.read_csv(r"C:\Users\user\Downloads\train.csv")
print(f"{df.head()}\n")


# region PREPROCESSING DATA

# 1) drop the obvious unrelated categorical features
df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis= 1, inplace= True)


# 2) handle NAs

# MISTAKE: THIS DROPS LOTS OF ROWS CAUSE THE DATASET IS FULL OF NAs
# df.dropna(inplace= True)

# so instead fill the NAs with values
df["Age"] = df["Age"].fillna(df["Age"].median())      # we fill with median and not the mean cause median will use the middle value (robust to outliers)
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0]) # get mode() (the most common values) then if there is a tie in values take the first one (5od el awal mesh el awal mokarar)


# 3) Feature engineering

# now for the one hot encoding for the gender
# MISTAKE : THIS WILL GENERATE TWO MORE FEATURES ONE OF THEM WILL BE REDUNDANT CAUSE THEY REPRESENT THE SAME THING NO MALE WILL BE FEMALE AT THE SAME TIME
# gender_column = pd.get_dummies(df["Sex"]).astype(int)

# make it binary encoding instead for to make the model less complex
male_column = pd.get_dummies(df["Sex"], drop_first= True, prefix= "Is").astype(int)
embarked_column = pd.get_dummies(df["Embarked"], prefix= "EmbarkedAt")
df = pd.concat([df, male_column, embarked_column], axis = 1)

# now we can get rid of the sex and embarked columns
df.drop(["Sex", "Embarked"], axis = 1, inplace= True)


# 4) check for correlations and handle redundant features
# now for the correlation to know which ones are redundant 
print(df.corr())

# THIS SHOWED THAT [PCLASS, ISMALE, FARE] ARE THE ONLY ONES CORRELATED WITH SURVIVED
# now check for the corr between them to remove ones that very corr with each other
# well the pclass is only corr with ismale by 0.15 so we can take both for now
# pclass and fare are highly corr with each other by 0.55 we can use only one of them to make things simpler
df.drop(["Age", "SibSp", "Parch", "Fare", "EmbarkedAt_S", "EmbarkedAt_C", "EmbarkedAt_Q"], axis = 1, inplace= True)


print(f"\n{df.head()}")
# endregion

# region DEFINING FUNCTIONS

# 1) creating the traintestsplit fn
# the one from scikit learn has multiple features 
# it allows u to pick urself if u want to declare the test size or train size 
# if u dont declare them it will automatically make test size = 25% of the whole dataset
# another features it to decide whether u want it to shuffle randomly and based on a pseudorandom number generator or not

# to do so we declare parameters values = none so that the user isnt obliged to input them
def train_test_split(train_size = None, test_size = None, shuffle = None, random_state = None):

    # now handle the default actions
    if (train_size == None & test_size == None):
        if(shuffle == True):
            if(random_state != None):
                return df.abs
