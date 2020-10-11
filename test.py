#coding=gbk
from scipy.stats import pearsonr
import pandas as pd
from sklearn.tree import DecisionTreeClassifier #Introduce decision tree model function
from sklearn.ensemble import BaggingClassifier #Introduce bagging model function
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler#Data standardization
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GridSearchCV#Grid search cross validation, used to select the best parameters
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier#Introduce KNN model function
from sklearn.model_selection import cross_val_score#Cross-validation

scaler = StandardScaler()

class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=20, out_features=400)
        self.fc3 = nn.Linear(in_features=400, out_features=2)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x
#DT tree
def get1():
    col_name = ['gameId', 'creationTime', 'gameDuration', 'seasonId', 'winner', 'firstBlood', 'firstTower',
                'firstInhibitor', 'firstBaron', 'firstDragon', 'firstRiftHerald', 't1_towerKills',
                't1_inhibitorKills', 't1_baronKills', 't1_dragonKills', 't1_riftHeraldKills', 't2_towerKills',
                't2_inhibitorKills', 't2_baronKills', 't2_dragonKills', 't2_riftHeraldKills']
    data = pd.read_csv("C:\\Users\\YUYAN\\Desktop\\Bigdata\\project_1\\test_set.csv",header=None,names=col_name)
    new_data = pd.read_csv("C:\\Users\\YUYAN\\Desktop\\Bigdata\\project_1\\new_data.csv",header=None,names=col_name)
    data = data.iloc[1:]
    new_data = new_data.iloc[1:]
    feature_cols = ['gameId','creationTime','gameDuration','seasonId','firstBlood','firstTower',
                'firstInhibitor','firstBaron','firstDragon','firstRiftHerald','t1_towerKills',
                't1_inhibitorKills','t1_baronKills','t1_dragonKills','t1_riftHeraldKills','t2_towerKills',
                't2_inhibitorKills','t2_baronKills','t2_dragonKills','t2_riftHeraldKills']
    X_train = data[feature_cols]
    X_train = scaler.fit_transform(X_train.astype(float))
    X_test = new_data[feature_cols]
    X_test = scaler.fit_transform(X_test.astype(float))
    y_label = data.winner
    Y = new_data.winner

    return X_train, X_test,y_label,Y

def get2():
    col_name = ['gameId', 'creationTime', 'gameDuration', 'seasonId', 'winner', 'firstBlood', 'firstTower',
                'firstInhibitor', 'firstBaron', 'firstDragon', 'firstRiftHerald', 't1_towerKills',
                't1_inhibitorKills', 't1_baronKills', 't1_dragonKills', 't1_riftHeraldKills', 't2_towerKills',
                't2_inhibitorKills', 't2_baronKills', 't2_dragonKills', 't2_riftHeraldKills']
    data1 = pd.read_csv("C:\\Users\\YUYAN\\Desktop\\Bigdata\\project_1\\test_set.csv")
    new_data1 = pd.read_csv("C:\\Users\\YUYAN\\Desktop\\Bigdata\\project_1\\new_data.csv")
    X_train_ann = data1.drop(['winner'], axis=1).values
    X_test_ann = new_data1.drop(['winner'], axis=1).values
    y_label_ann = data1['winner'].values
    Y_ANN = new_data1['winner'].values
    """for j in range(16):
        for i in range(20586):
            X_train_ann[i, j] = (X_train_ann[i, j] - np.min(X_train_ann[:, j])) / (np.max(X_train_ann[:, j]) - np.min(X_train_ann[:, j]))
            X_test_ann[i, j] = (X_test_ann[i, j] - np.min(X_test_ann[:, j])) / (np.max(X_test_ann[:, j]) - np.min(X_test_ann[:, j]))"""
    X_train_ann = scaler.fit_transform(X_train_ann)
    X_train_ann = torch.FloatTensor(X_train_ann)
    X_test_ann = scaler.fit_transform(X_test_ann)
    X_test_ann = torch.FloatTensor(X_test_ann)
    y_label_ann = torch.LongTensor(y_label_ann) - 1
    Y_ANN = torch.LongTensor(Y_ANN) - 1
    return X_train_ann, X_test_ann,y_label_ann,Y_ANN

def test_dt():
    accuracy_DT1 = []
    accuracy_DT2 = []
    for i in range(5,15,1):
        X_train, X_test, y_label, Y = get1()
        DT_clf = DecisionTreeClassifier(max_depth=i, random_state=0, criterion='entropy')
        DT_clf = DT_clf.fit(X_train, y_label)
        y_predict = DT_clf.predict(X_test)
        accuracy_DT1.append(accuracy_score(Y, y_predict))
    max_acc1 = max(accuracy_DT1)
    best_depth1 = accuracy_DT1.index(max_acc1)+5
    print("Under entropy,max accuracy is ", max_acc1, "\t The max_depth is ", best_depth1)
    for i in range(5,15,1):
        X_train, X_test, y_label, Y = get1()
        #“gini” and “entropy” can be chosen for “criterion”.
        # “max_depth” is to set the max depth for the decision tree.
        DT_clf = DecisionTreeClassifier(max_depth=i, random_state=0, criterion='gini') #Instantiate decision tree model
        DT_clf = DT_clf.fit(X_train, y_label)# put features and labels into classifier model go on fitting
        y_predict = DT_clf.predict(X_test)
        accuracy_DT2.append(accuracy_score(Y, y_predict))
    x = range(5,15,1)
    max_acc2 = max(accuracy_DT2)
    best_depth2 = accuracy_DT2.index(max_acc2) +5
    print("Under gini,the max accuracy is ",max_acc2,"\t The max_depth is ",best_depth2)
    y1 = accuracy_DT1
    y2 = accuracy_DT2
    #Draw an image of the influence of parameters on accuracy
    plt.figure(num='DT_parameters', figsize=(10, 10), dpi=75, facecolor='#FFFFFF', edgecolor='#0000FF')
    plt.plot(x,y1,'r-o',label = 'criterion=entropy')
    plt.plot(best_depth1, max_acc1, 'ks')
    plt.plot(x,y2,'b-o',label = 'criterion=gini')
    plt.plot(best_depth2, max_acc2, 'ks')
    show_max1 = '[' + str(best_depth1) + ', ' + str(max_acc1) + ']'
    show_max2 = '[' + str(best_depth2) + ', ' + str(max_acc2) + ']'
    #Mark the points in the graph
    plt.annotate(show_max1, xytext=(7, max_acc1+0.0005), xy=(best_depth1, max_acc1),arrowprops=dict(arrowstyle='-|>',connectionstyle='arc3',color='k'))
    plt.annotate(show_max2, xytext=(7, max_acc2+0.0005), xy=(best_depth2, max_acc2),arrowprops=dict(arrowstyle='-|>',connectionstyle='arc3',color='k'))
    plt.xticks(x)
    #Mark the X, Y axis
    plt.xlabel('max_depth')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')
    plt.show()

def test_ann():
    accuracy_ANN = []
    X_train_ann, X_test_ann, y_label_ann, Y_ANN = get2()
    for i in range(2):
        model = ANN()
        # Cross loss function, used for classification
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=(0.1)**(i+1))
        epochs = 100
        loss_arr = []
        for i in range(epochs):
            y_hat = model.forward(X_train_ann)
            loss = criterion(y_hat, y_label_ann)
            loss_arr.append(loss)
            optimizer.zero_grad()#set the gradient to zero
            loss.backward() # Back propagation calculates the gradient
            optimizer.step()# Use optimizer to update parameters
        predict_out = model(X_test_ann)
        a, predict_y = torch.max(predict_out, 1)
        accuracy_ANN.append(accuracy_score(predict_y, Y_ANN))
    max_acc = max(accuracy_ANN)
    best_rate = (0.1)**(accuracy_ANN.index(max_acc) + 1)
    print("The max accuracy is ", max_acc, "\t The best_rate is ", best_rate)

def test_knn():
    X_train_knn, X_test_knn, y_label_knn, Y_kNN = get1()
    #Frame the parameter range
    n_neighbors = list(range(1, 8))
    weight_options = ['uniform', 'distance']
    algorithm_options = ['auto', 'ball_tree', 'kd_tree', 'brute']
    leaf_size_option = list(range(1, 50, 10 ))
    #Integrate parameter ranges into the grid
    param_grid = dict(n_neighbors=n_neighbors, weights=weight_options, algorithm=algorithm_options,leaf_size = leaf_size_option)
    cls_knn = KNeighborsClassifier(n_neighbors=4)
    gridKNN = GridSearchCV(cls_knn, param_grid, cv=5, scoring='accuracy', verbose=1)#Grid function instantiation
    gridKNN.fit(X_train_knn, y_label_knn)#model fit
    print(gridKNN.best_params_)
    cls_knn = KNeighborsClassifier(algorithm='auto', leaf_size=1, metric='minkowski', metric_params=None, n_jobs=None,
                                   n_neighbors=5, p=2, weights='distance')
    cls_knn.fit(X_train_knn, y_label_knn)
    y_predict = cls_knn.predict(X_test_knn)
    a = accuracy_score(y_predict, Y_kNN)#Model evaluation
    print(a)

def selection_feature():
    col_name = ['gameId', 'creationTime', 'gameDuration', 'seasonId', 'winner', 'firstBlood', 'firstTower',
                'firstInhibitor', 'firstBaron', 'firstDragon', 'firstRiftHerald', 't1_towerKills',
                't1_inhibitorKills', 't1_baronKills', 't1_dragonKills', 't1_riftHeraldKills', 't2_towerKills',
                't2_inhibitorKills', 't2_baronKills', 't2_dragonKills', 't2_riftHeraldKills']
    data = pd.read_csv("C:\\Users\\YUYAN\\Desktop\\Bigdata\\project_1\\test_set.csv")
    data = data.iloc[1:].values
    r = []
    b = data[:, 4]
    for i in range(20):
        c,d = pearsonr(b, data[:, i])#Calculate correlation coefficient
        r.append(d)
        print("The Ccorrelation coefficient between winner and ", col_name[i], "is：\n", r[i])

#Run the function
selection_feature()
test_knn()
test_ann()
test_dt()