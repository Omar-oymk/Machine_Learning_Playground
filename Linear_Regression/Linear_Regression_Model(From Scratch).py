import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# print(df.describe())
df = pd.read_csv("Grades.csv")
print(df.info())
print(df.head(8))
df.drop("Name", axis= 1, inplace= True)

# region manual
# global variables
w, b = 0, 0
no_of_epochs = 10000
learning_rate = 0.01
datapoints, features = df.shape
x = np.array(df["Hours_Studied"])

def predict(value):           # predict on one datapoint
    yhat = value*w + b
    return yhat

def Mean_Squared_Error(y, yhat):
    total = 0
    
    total = (y-yhat)**2
    total = np.sum(total)
    mse = (1/datapoints) * total
    
    return mse

def backward_pass(y, yhat):
    '''return by order (gradient(w), gradient(b))'''
    total_w = ((df['Hours_Studied']).to_numpy())*(y-yhat)
    total_b = (y-yhat)

    total_w = np.sum(total_w)
    total_b = np.sum(total_b)

    return (-(1/datapoints)*2*total_w), (-(1/datapoints)*2*total_b)

def gradient_descent(y, yhat):
    gradient_w, gradient_b = backward_pass(y, yhat)
    return (w - learning_rate* gradient_w), (b - learning_rate* gradient_b)

def r2_score(y, yhat):
    """Compute the R² score."""
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)

def train():
    global w, b 
    for epoch in range(no_of_epochs):
        
        yhat = predict(x)
        
        if (epoch%1000 == 0):
            print(Mean_Squared_Error(df["Final_Score"], yhat))
            # plt.scatter(df["Hours_Studied"], df['Final_Score'], color = 'red', label = 'Actual Values')
            # plt.plot(df['Hours_Studied'], yhat, color = 'blue', label = 'Predicted line')
            # plt.xlabel("Hours Studied")
            # plt.ylabel("Final Score")
            # plt.legend()
            # plt.grid()
            # plt.show()
        
        w, b = gradient_descent(df["Final_Score"], yhat)
    final_yhat = predict(x)
    print("Final R² =", r2_score(df["Final_Score"], final_yhat))

    
train()
#endregion

print("Final score manually (Hours=5) =", predict(5))
# scikit-learn
model = LinearRegression()
X_train, X_test, Y_train, Y_test = train_test_split(df["Hours_Studied"], df["Final_Score"], test_size= 0.2, random_state=42)
model.fit(X_train.values.reshape(-1, 1), Y_train.values.reshape(-1, 1))

print("Final score scikitlearn (Hours=5) =", model.predict([[5]])[0][0])
print("The R^2 Score: " + str(r2_score(Y_test.values.reshape(-1, 1), model.predict(X_test.values.reshape(-1, 1)))))