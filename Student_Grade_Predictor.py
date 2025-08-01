from os import system
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv(r"c:\Users\user\Desktop\Study\Codes\Artificial intelligence & Machine learning\Projects\Student Grade Predictor (Simple Linear Regression)\Grades.csv")
print(df.sample(10))

# preprocess data
# df.drop(columns=['Unnamed: 3'], inplace= True)
# print(df.sample(5))
# df.to_csv(r"c:\Users\user\Desktop\Study\Codes\Artificial intelligence & Machine learning\Projects\Student Grade Predictor (Simple Linear Regression)\Grades.csv", index=False)

# describe the data
print("\n")
print(df.describe())

# train the model
X = df['Hours_Studied'].to_numpy()
Y = df['Final_Score'].to_numpy()

model = LinearRegression()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.3, random_state= 42)
model.fit(X_train.reshape(-1, 1), Y_train.reshape(-1, 1))

# plot the result
plt.scatter(X_train, Y_train, color = 'blue')
plt.plot(X_train, model.predict(X_train.reshape(-1, 1)), color = 'red')
plt.title("Predictions vs Actual")
plt.xlabel("Hours Studied")
plt.ylabel("Final Score")
plt.grid()
plt.show()

# it shows that it is nearly accurate an acceptable accuracy

# NOW LETS MAKE THE APP
def Grade_evaluate(score):
    if score <= 50:
        return 'F'
    elif score <=60:
        return 'D'
    elif score <= 70:
        return 'C'
    elif score <= 80:
        return 'B'
    else:
        return 'A'
    
system("cls")       # clears console
print(r'''                     .;                                   .              .-.                           .-.                                           .             
.;.       .-.       .;'                               ...;...     .;;.`-'                    .'       (_) )-.                   .'   .-.         ...;...       .-. 
  `;     ;' .-.    .;  .-.   .-.  . ,';.,';.  .-.      .'.-.     ;; (_;    .;.::..-.    .-..'  .-.      .:   \  .;.::..-.  .-..'     `-' .-.      .'.-.  .;.::.`-' 
   ;;    ;.;.-'   ::  ;     ;   ;';;  ;;  ;;.;.-'    .; ;   ;'  ;;         .;   ;   :  :   ; .;.-'     .:'    ) .;  .;.-' :   ;     ;'  ;       .; ;   ;'.;   .-.  
  ;;  ;  ;;`:::'_;;_.-`;;;;'`;;' ';  ;;  ';  `:::' .;   `;;'   ;;    `;;'.;'    `:::'-'`:::'`.`:::'  .-:. `--'.;'    `:::'`:::'`._.;:._.`;;;;'.;   `;;'.;'    `-'  
  `;.' `.;'                     _;        `-'                  `;.___.'                             (_/                                                            
''')

Input = float(input("Enter Hours of study: "))
Input = np.array([round(Input, 2)])
Final_Score = model.predict(Input.reshape(-1, 1))
print("Final Score = " + str(round(Final_Score[0][0], 2)))
print("Grade = " + Grade_evaluate(Final_Score[0][0]))
print("The mean absolute error of this model is: " + str(round(mean_absolute_error(Y_test.reshape(-1, 1), model.predict(X_test.reshape(-1, 1))), 3)))