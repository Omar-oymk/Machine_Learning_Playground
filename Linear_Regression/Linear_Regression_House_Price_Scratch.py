import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load dataset
df = pd.read_csv(r"C:\Users\user\Downloads\archive (1)\house_price_regression_dataset.csv")
print(df.head())
print(f"\n{df.info()}")

#region preprocessing
## use scaling (using standardization)

def ZScoreStandardization(X):
    return (X - X.mean()) / X.std()

## correlation to remove redundant features
sns.heatmap(df.corr().abs(), cmap="coolwarm")
plt.show()

### from that we can see that it is highly with lot_size, garage_size and year_built
### however the lot_size is kinda highly corr with year_built so we will just use the lot_size for the sake of simplicity

# X = df[["Lot_Size", "Garage_Size"]]
X = df[["Lot_Size"]]
X = ZScoreStandardization(X)
Y = df["House_Price"]
Y_mean = Y.mean()
Y_std = Y.std()
Y = ZScoreStandardization(Y)

# now we split the data into training and testing data
def train_test_split(X, Y, train_size = None, test_size = None, shuffle = False, random_state = 0):
    if(train_size is not None and test_size is not None): raise ValueError("Cannot define both test and train size at the same time")
    if(train_size is not None): 
        if(0 <= train_size <= 1): test_size = 1 - train_size
        else: raise ValueError("train_size has to be between 0 and 1 inclusive")
    elif(test_size is not None): 
        if(0 <= test_size <= 1): train_size = 1 - test_size
        else: raise ValueError("train_size has to be between 0 and 1 inclusive")
    else:
        train_size = 0.75
        test_size = 0.25

    n = (int)(train_size * len(X))
    if (shuffle):
        np.random.seed(random_state)
        indices_shuffled = np.random.permutation(len(X))
        shuffled_x = X.iloc[indices_shuffled]
        shuffled_y = Y.iloc[indices_shuffled]
        x_train, x_test, y_train, y_test = shuffled_x[:n], shuffled_x[n:], shuffled_y[:n], shuffled_y[n:]
    else:
        x_train, x_test, y_train, y_test = X[:n], X[n:], Y[:n], Y[n:]

    return x_train, x_test, y_train, y_test
#endregion

#region Model Training

class LinearRegression:
    """VISUALIZATION WORKS ONLY IF X HAS ONLY 1 FEATURE"""
    def __init__(self, epochs = 10000, learning_rate = 0.001, verbose = False, visualization = False):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.visualization = visualization
        self.coef_ = None
        self.bias_ = None

    def fit(self, x_train, y_train):
        x_columns = x_train.columns
        label = y_train.name
        x_train = x_train.values
        y_train = y_train.values
        self.coef_, self.bias_ = self._initialize_weights(x_train)

        if(x_train.shape[1] != 1): print("VISUALIZTION WORKS ON ONLY 1 FEATURE")
        for i in range(self.epochs):
            if (i % 1000 == 0): 
                if(self.verbose):
                    print("---------------------------------------------")
                    print(f"MSE = {self._cost_function(x_train, y_train)}")
                    print(f"RMSE = {np.sqrt(self._cost_function(x_train, y_train))}")
                    print("---------------------------------------------")
                if(self.visualization):
                    if(x_train.shape[1] == 1):
                        plt.title("Training Progress")
                        plt.scatter(x_train, y_train, color = "red", label = "True values")
                        plt.plot(x_train, self._linear(x_train), "b--", label = "Model Fitting")
                        plt.xlabel(x_columns[0])
                        plt.ylabel(label)
                        plt.legend()
                        plt.show()
            self._gradient_descent(x_train, y_train)


    def _initialize_weights(self, x_train):
        self.coef_ = np.zeros(x_train.shape[1])
        self.bias_ = 0
        return self.coef_, self.bias_

    def _linear(self, x_train):
        return np.dot(x_train, self.coef_) + self.bias_

    def _cost_function(self, x_train, y_train):
        # mse fn 
        return (1/x_train.shape[0]) * np.sum((self._linear(x_train) - y_train)**2)

    def _gradient(self, x_train, y_train):
        # dCostdWeights = -(1/x_train.shape[0]) * 2 * np.sum(x_train * (self._Linear(x_train) - y_train))     # this will only work if it were for 1 feature
        dCostdWeights = (1/x_train.shape[0]) * 2 * np.dot(x_train.T, (self._linear(x_train) - y_train))
        dCostdBias = (1/x_train.shape[0]) * 2 * np.sum((self._linear(x_train) - y_train))
        return dCostdWeights, dCostdBias
    
    def _gradient_descent(self, x_train, y_train):
        dCostWeights, dcostBias = self._gradient(x_train, y_train)
        self.coef_ = self.coef_ - self.learning_rate * dCostWeights
        self.bias_ = self.bias_ - self.learning_rate * dcostBias

    def predict(self, x_test):
        return self._linear(x_test) * Y_std + Y_mean
    
#endregion

#region main.py
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, shuffle = True, random_state = 42)
model = LinearRegression(visualization= True)
model.fit(x_train, y_train)
print(model.predict(x_test))
#endregion