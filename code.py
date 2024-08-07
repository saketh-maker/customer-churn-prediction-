#Importing necessary Libraries
import numpy as np
import pandas as pd
import tensorflow as tf

data = pd.read_csv("Churn_Modelling.csv")
data.describe()

data.head()#printing the head values to know adout the data

X = data.iloc[:,3:-1].values
print(X)#alloting the x values to find

# Dependent Variable Vectors
Y = data.iloc[:,-1].values
# print(Y)
# print("______________________________________________________________")

#Encoding Categorical Variable Gender
from sklearn.preprocessing import LabelEncoder
LE1 = LabelEncoder()
X[:,2] = np.array(LE1.fit_transform(X[:,2]))

#Encoding Categorical variable Geography
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct =ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder="passthrough")
X = np.array(ct.fit_transform(X))

#Splitting dataset into training and testing dataset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

#Performing Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Early Stoppage Regularization
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

#Initialising Ann for perdicting the output
ann = tf.keras.models.Sequential()

 #Adding the First Hidden Layer with relu activation function
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))

 #Adding the Second Hidden Layer with relu activiation function
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))

 #Adding Output Layer with sigmoid function
ann.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))

#Compiling ANN
ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])#change the optimzer and stooping use drop out

#Fitting ANN
ann.fit(X_train,Y_train,batch_size=32,epochs = 10)

#Predicting result for Single Observation
print(ann.predict(sc.transform([[619, 0, 0, 600, 1, 40, 3, 100000, 2, 1, 1,56000]])) > 0.4)


                                #CLICK ON OUTPUTFILE TO SEE OUTPUT#
