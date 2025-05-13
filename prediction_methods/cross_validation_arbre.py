from pandas import DataFrame

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

import scipy as sp

import numpy as np


def cross_validation_arbre_classification(data: DataFrame) -> int:
    x = np.array(data["timestamp"], data["Valeur"])
    y = data["alerte"]
    k = 5 
    kf = KFold(n_splits=k, shuffle=False, random_state=None)

    depth_array = np.arange(1,20)
    estd_accuracy = []

    for nbdepth in depth_array:
        clf = DecisionTreeClassifier(max_depth=nbdepth,criterion='gini')
        scores = cross_val_score(clf, x, y, cv=kf)
        estd_accuracy.append(scores.mean())
        
    print(estd_accuracy)

    max_accuracy = 0
    index_max = 0
    for i in range(estd_accuracy):
        if max_accuracy < estd_accuracy[i]:
            max_accuracy = estd_accuracy[i]
            index_max = i

    plt.plot(depth_array,estd_accuracy)
    plt.xlabel("Max depth of the tree")
    plt.ylabel("Classification accuracy")
    plt.grid()
    plt.show()

    return depth_array[index_max]



def cross_validation_arbre_regression(data: DataFrame) -> int:
    x = data["timestamp"]
    y = data["Valeur"]
    k = 5 
    kf = KFold(n_splits=k, shuffle=False, random_state=None)

    depth_array = np.arange(1,20)
    reg_MSE=[]

    for N in depth:
        regrN = DecisionTreeRegressor(max_depth=N) 
        mserr=[]
        for train_index, test_index in kf.split(x):
            regrN = regrN.fit(x[train_index], y[train_index])
            y_pred = regrN.predict(x[test_index]); 
            y_t = y[test_index].ravel() # to force same dimensions as those of y_pred
            mserr.append(np.square(y_t - y_pred).sum())
        
        reg_MSE.append(np.asarray(mserr).mean())
    print(reg_MSE)
    min_MSE = reg_MSE[0]
    index_opti = 0
    for i in range(reg_MSE):
            if min_MSE > reg_MSE[i]:
                max_amin_MSEccuracy = reg_MSE[i]
                index_otpi = i

    plt.plot(depth, reg_MSE)
    plt.xlabel('max Depth of the reg_tree')
    plt.ylabel ('MSE')
    plt.grid()
    plt.show()

    return depth_array[index_opti]

    