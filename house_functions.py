#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 14:44:04 2018

@author: hanzhang
"""
import numpy as np
import matplotlib.pyplot as plt

#plotting functions 

def plot_scatter(pred,y,title):
    f, ax = plt.subplots(1, 2, figsize=(8, 4))
    #ax.scatter(np.e**(val_y), np.e**(Y_pred), edgecolors=(0, 0, 0))
    ax[0].scatter((pred), (y), edgecolors=(0, 0, 0))
    ax[0].set_xlabel('predictions')
    ax[0].set_ylabel('data')
    ax[0].set_title(title)
    
    ax[1].scatter(pred,(pred-y),edgecolors=(0,0,0))
    ax[1].set_xlabel('pred')
    ax[1].set_ylabel('predictions-data')
    #a2.set_title(title)
    
    plt.title(title)
    
    

from sklearn.model_selection import learning_curve
#from sklearn.model_selection import ShuffleSplit

# plot learning curve 
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=10, train_sizes=np.linspace(.1, 1.0, 6)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

from sklearn.metrics import make_scorer 
# Define a function to calculate RMSE
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true-y_pred)**2))

# Define a function to calculate negative RMSE (as a score)
def nrmse(y_true, y_pred):
    return -1.0*rmse(y_true, y_pred)

neg_rmse = make_scorer(nrmse)

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error

def print_model_results(model,name,cv,y,pred,X):
    
    print('RMSE',rmse(y,pred))
    #print('Score',model.score(X,y))
    result_xgb = cross_val_score(model, X, y, cv=cv,scoring=neg_rmse)
    print('Cross Validation '+name,' ',np.mean(-result_xgb))
    return np.mean(-result_xgb)

# remove two outliers 
#train_std.head()
#data_dummy=data_dummy.drop(data_dummy.index[[523,1298]],axis=0)
#data_loc.drop(data.index[[523,1298]], inplace=True,axis=0)
