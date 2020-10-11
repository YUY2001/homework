#coding=gbk
import pandas as pd
from sklearn.tree import DecisionTreeClassifier #Introduce decision tree model function
from sklearn.ensemble import BaggingClassifier#Introduce bagging model function
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler#Data standardization
import numpy as np
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier #Introduce KNN model function
from sklearn.model_selection import cross_val_score #Cross-validation
#Used to visualize decision trees
import graphviz
import pydotplus
from sklearn import tree
from IPython.display import Image
import os
scaler = StandardScaler()
# Get and format the data for Decision tree and bagging classifier
col_name = ['gameId', 'creationTime', 'gameDuration', 'seasonId', 'winner', 'firstBlood', 'firstTower',
                'firstInhibitor', 'firstBaron', 'firstDragon', 'firstRiftHerald', 't1_towerKills',
                't1_inhibitorKills', 't1_baronKills', 't1_dragonKills', 't1_riftHeraldKills', 't2_towerKills',
                't2_inhibitorKills', 't2_baronKills', 't2_dragonKills', 't2_riftHeraldKills']
feature_cols = ['gameId', 'creationTime', 'gameDuration', 'seasonId', 'firstBlood', 'firstTower',
                'firstInhibitor', 'firstBaron', 'firstDragon', 'firstRiftHerald', 't1_towerKills',
                't1_inhibitorKills', 't1_baronKills', 't1_dragonKills', 't1_riftHeraldKills', 't2_towerKills',
                't2_inhibitorKills', 't2_baronKills', 't2_dragonKills', 't2_riftHeraldKills']
def get1():
    data = pd.read_csv("C:\\Users\\YUYAN\\Desktop\\Bigdata\\project_1\\test_set.csv",header=None,names=col_name)
    new_data = pd.read_csv("C:\\Users\\YUYAN\\Desktop\\Bigdata\\project_1\\new_data.csv",header=None,names=col_name)
    data = data.iloc[1:]
    new_data = new_data.iloc[1:]
    #Separate features and labels
    X_train = data[feature_cols]
    X_train = scaler.fit_transform(X_train.astype(float)) #Standardize data processing
    X_test = new_data[feature_cols]
    X_test = scaler.fit_transform(X_test.astype(float))  #Standardize data processing
    y_label = data.winner
    Y = new_data.winner
    return X_train, X_test,y_label,Y # Classify the features and labels of the data set, and return their values

# Get and format the data for Ann classifier
def get2():
    data1 = pd.read_csv("C:\\Users\\YUYAN\\Desktop\\Bigdata\\project_1\\test_set.csv")
    new_data1 = pd.read_csv("C:\\Users\\YUYAN\\Desktop\\Bigdata\\project_1\\new_data.csv")
    # Separate features and labels
    X_train_ann = data1.drop(['seasonId'], axis=1).values
    X_test_ann = new_data1.drop(['seasonId'], axis=1).values
    y_label_ann = data1['winner'].values
    Y_ANN = new_data1['winner'].values
    """for j in range(20):
        for i in range(20586):
            X_train_ann[i, j] = (X_train_ann[i, j] - np.min(X_train_ann[:, j])) / (np.max(X_train_ann[:, j]) - np.min(X_train_ann[:, j]))
            X_test_ann[i, j] = (X_test_ann[i, j] - np.min(X_test_ann[:, j])) / (np.max(X_test_ann[:, j]) - np.min(X_test_ann[:, j]))"""
    # convert split data from Numpy arrays to PyTorch tensors
    X_train_ann = scaler.fit_transform(X_train_ann)
    X_train_ann = torch.FloatTensor(X_train_ann)
    X_test_ann = scaler.fit_transform(X_test_ann)
    X_test_ann = torch.FloatTensor(X_test_ann)
    y_label_ann = torch.LongTensor(y_label_ann) - 1
    Y_ANN = torch.LongTensor(Y_ANN) - 1
    return X_train_ann, X_test_ann,y_label_ann,Y_ANN # Classify the features and labels of the data set, and return their values

#building decision tree model
def model_dt():
    start1 = time.process_time() #Get start time
    X_train, X_test, y_label, Y = get1()
    # “gini” and “entropy” can be chosen for “criterion”.
    # “max_depth” is to set the max depth for the decision tree.
    DT_clf = DecisionTreeClassifier(max_depth = 8,random_state = 0,criterion='entropy') #Instantiate decision tree model
    DT_clf = DT_clf.fit(X_train, y_label)# put features and labels into classifier model go on fitting
    y_predict = DT_clf.predict(X_test)
    a = accuracy_score(Y, y_predict)
    end1 = time.process_time() #Get end time
    b = end1 - start1 #Get training time
    #Visual decision tree
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/' #Get absolute path
    dot_tree = tree.export_graphviz(DT_clf, out_file=None, feature_names=feature_cols,class_names=['winner','losser'], filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_tree)
    img = Image(graph.create_png())
    graph.write_png("out.png") #score picture
    print("The accuracy of DT is", a, "\tThe running time of DT is", b)

#build ANN model
def model_ann():
    start1 = time.process_time() #Get start time
    X_train_ann, X_test_ann, y_label_ann, Y_ANN = get2()
    # create a neural networks
    class ANN(nn.Module):
        def __init__(self):
            super().__init__()
            # 20 means 20 features
            self.fc1 = nn.Linear(in_features=20, out_features=400)
            # 2 means output 2 labels
            self.fc3 = nn.Linear(in_features=400, out_features=2)
        def forward(self, x):
            x = torch.tanh(self.fc1(x)) # use tanh for activation function
            x = self.fc3(x)
            x = F.softmax(x,dim = 1)
            return x

    model = ANN()
    criterion = nn.CrossEntropyLoss() # use cross entropy loss to compute loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    epochs = 100
    loss_arr = []
    for i in range(epochs):
        y_hat = model.forward(X_train_ann)
        loss = criterion(y_hat, y_label_ann)
        loss_arr.append(loss)
        """if i % 10 == 0:
            print(f'Epoch: {i} Loss: {loss}')"""
        optimizer.zero_grad()  #set the gradient to zero
        loss.backward() # Back propagation calculates the gradient
        optimizer.step() # Use optimizer to update parameters
    predict_out = model(X_test_ann)
    a, predict_y = torch.max(predict_out, 1)
    c = accuracy_score(predict_y, Y_ANN)#Model evaluation
    end1 = time.process_time()#Get end time
    b = end1 - start1
    print("The accuracy of ANN is", c, "\tThe running time of ANN is", b)

def KNN():
    start1 = time.process_time()
    X_train_nnn, X_test_nnn, y_label_nnn, Y = get2()
    # 'n_neighbors' is the number of neighbors to use by default for kneighbors queries.
    # “weights” can be chosen from ‘uniform’ which represents that all points in each neighborhood are weighted equally,
    # ’distance’ which denotes weight points by the inverse of their distance
    # callable’ which can be customize.
    # “algorithm” can be chosen from ‘auto’, ‘ball_tree’, ‘kd_tree’ and ‘brute’.
    # “leaf_size” means the size of leaf passed to BallTree or KD.
    cls_knn = KNeighborsClassifier(algorithm='auto', leaf_size=1, metric='minkowski', metric_params=None, n_jobs=None,
                                   n_neighbors=5, p=2, weights='distance') #Instantiate the KNN model
    cls_knn.fit(X_train_nnn, y_label_nnn)#Model training
    y_predict = cls_knn.predict(X_test_nnn)
    a = accuracy_score(y_predict, Y)#Model evaluation
    end1 = time.process_time()
    b = end1 - start1
    print("The accuracy of KNN is", a, "\tThe running time of KNN is", b)

#build bagging model
def bagging_dt():
    start1 = time.process_time()
    X_train, X_test, y_label, Y = get1()
    bag_clf = BaggingClassifier(DecisionTreeClassifier(max_depth = 8,random_state = 0,criterion='entropy'), n_estimators=200, max_samples=800, bootstrap=True, n_jobs=-1)#Instantiate the model
    bag_clf.fit(X_train, y_label)#Model training
    y_pred = bag_clf.predict(X_test)
    a = accuracy_score(y_pred, Y)#Model evaluation
    end1 = time.process_time()
    b = end1 - start1
    print("The accuracy of Bagging(DT) is", a, "\tThe running time of Bagging(DT) is" ,b)

def bagging_knn():
    start1 = time.process_time()
    X_train, X_test, y_label, Y = get1()
    bag_clf = BaggingClassifier(KNeighborsClassifier(algorithm='brute', leaf_size=1, metric='minkowski', metric_params=None,
                                          n_jobs=None, n_neighbors=5, p=2, weights='distance'),n_estimators=200, max_samples=800, bootstrap=True, n_jobs=-1)#Instantiate the model
    bag_clf.fit(X_train, y_label)#Model training
    y_pred = bag_clf.predict(X_test)
    a = accuracy_score(y_pred, Y)#Model evaluation
    end1 = time.process_time()
    b = end1 - start1
    print("The accuracy of Bagging(KNN) is", a, "\tThe running time of Bagging(KNN) is", b)

# Run function
model_ann()
model_dt()
KNN()
bagging_dt()
bagging_knn()
