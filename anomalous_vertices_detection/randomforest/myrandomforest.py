from distutils.command.config import config

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def randomforestcalling(vertices_output_path):

    #Importing Dataset
    dataset = pd.read_csv(vertices_output_path)

    #Preparing Data For Training
    X = dataset.iloc[:, 0:7].values
    Y = dataset.iloc[:, 8].values

    # print(X)
    # print(Y)

    X_train, X_test, y_train, y_test2 = train_test_split(X, Y, test_size=0.2, random_state=0)

    X_train2, X_test2, y_train2, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


    #Feature Scaling

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    count_n_estimators = [5,10,15,20,25,30]
    all_accuracy_score = []
    threshold = 0.8

    for ne in count_n_estimators:
        #  Training the Algorithm
        regressor = RandomForestRegressor(n_estimators=ne, random_state=0)
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)

        # print(y_test)
        # print(y_pred)

        y_pred = (y_pred>threshold).astype(int)

        # print(y_test)
        # print(y_pred)

        #Evaluating the Algorithm
        print('\nNo of Tress in the Forest is set to ',ne)
        # print(confusion_matrix(y_test,y_pred))
        # con = confusion_matrix(y_test,y_pred)

        unique_label = np.unique(y_test)
        print(pd.DataFrame(confusion_matrix(y_test, y_pred, labels=unique_label),
                           index=['True: {:}'.format(x) for x in ['Fake->0','Real->1']],
                           columns=['Pred: {:}'.format(x) for x in ['Fake->0','Real->1']]))

        # print(classification_report(y_test,y_pred))
        print("Average is ",accuracy_score(y_test,y_pred))

        all_accuracy_score.append(accuracy_score(y_test,y_pred))

    print('\nAvg Accuracy: ',(sum(all_accuracy_score) / len(all_accuracy_score)))

    randomforestplotcomparisiongraph(count_n_estimators,all_accuracy_score)


def randomforestplotcomparisiongraph(count_n_estimators, all_accuracy_score ):
    # print(count_n_estimators)
    # print(all_accuracy_score)

    plt.xlabel('No of Tress in the forest (n_estimators)')
    plt.ylabel('Accuracy in predication (accuracy_score)')
    plt.plot(count_n_estimators, all_accuracy_score, color='blue', linestyle='dashed', linewidth = 3,
         marker='o', markerfacecolor='red', markersize=12)
    plt.axis([0, 35, 0.9, 1])
    plt.show()
