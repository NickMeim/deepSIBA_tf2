from __future__ import division, print_function
import numpy as np
from numpy import inf, ndarray
import pandas as pd
import tensorflow as tf
import os
import random
import keras
import sklearn
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import metrics
import re
from keras import optimizers
from keras import losses
from keras import regularizers
import keras.backend as K
from keras.models import model_from_json
from keras.models import load_model, Model
from tempfile import TemporaryFile
from keras import layers
from keras.callbacks import History, ReduceLROnPlateau
from keras.layers import Input, BatchNormalization, Activation
from keras.layers import Dense, Bidirectional, Dropout, Layer
from keras.initializers import glorot_normal
from keras.regularizers import l2
from functools import partial
from multiprocessing import cpu_count, Pool
from keras.utils.generic_utils import Progbar
from copy import deepcopy
from math import ceil
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error


#Define custom metrics for evaluation
def r_square(y_true, y_pred):
    from keras import backend as K
    (y_pred, sigma) = tf.split(y_pred, num_or_size_splits=[-1, 1], axis=-1)
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))

def get_cindex(y_true, y_pred):
    (y_pred, sigma) = tf.split(y_pred, num_or_size_splits=[-1, 1], axis=-1)
    g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

    f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
    f = tf.linalg.band_part(tf.cast(f, tf.float32), -1, 0)

    g = tf.reduce_sum(tf.multiply(g, f))
    f = tf.reduce_sum(f)

    return tf.where(tf.equal(g, 0), 0.0, g/f)

def pearson_r(y_true, y_pred):
    (y_pred, sigma) = tf.split(y_pred, num_or_size_splits=[-1, 1], axis=-1)
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return K.mean(r)

def custom_mse(y_true,y_pred):
    (y_pred, sigma) = tf.split(y_pred, num_or_size_splits=[-1, 1], axis=-1)
    er=K.mean(K.square(y_pred - y_true), axis=-1)
    return er

def mse_sliced(th):
    def mse_similars(y_true,y_pred):
        (y_pred, sigma) = tf.split(y_pred, num_or_size_splits=[-1, 1], axis=-1)
        condition = tf.math.less_equal(y_pred,th)
        indices = tf.where(condition)
        slice_true = tf.gather_nd(y_true,indices)
        slice_pred = tf.gather_nd(y_pred,indices)
        mse_sliced = K.mean(K.square(slice_pred - slice_true), axis=-1)
        return mse_sliced
    return mse_similars

#Model evaluation function
def model_evaluate(y_pred,Y_cold,thresh,df_cold):
    true = np.reshape(Y_cold,len(df_cold))
    pred = np.reshape(y_pred,len(df_cold))
    cor = np.corrcoef(true,pred)
    mse_all = sklearn.metrics.mean_squared_error(true,pred)
    # calculate mse of similars
    if (len(pred[np.where(pred<=thresh)])>0):
        mse_sims = sklearn.metrics.mean_squared_error(true[pred<=thresh],pred[pred<=thresh])
    else:
        mse_sims = "None"
    # turn to categorical to calculate precision and accuracy
    true_cat = true <= thresh
    pred_cat = pred <= thresh
    pos = np.sum(pred_cat)
    # calculate accuracy and fpr and precision
    tn, fp, fn, tp=confusion_matrix(true_cat,pred_cat).ravel() #see sklearn.metrics.confusion_matrix documentation
    if (len(pred[np.where(pred<=thresh)])>0):
        prec = tp/(fp+tp)
    else: 
        prec = "None"
    fpr=fp/(fp+tn)
    acc = (tp+tn)/(tp+tn+fp+fn)
    result =pd.DataFrame({'cor' : cor[0,1], 'mse_all' : mse_all, 'mse_similars' : mse_sims,'precision': prec, 'accuracy': acc,
                         'FPR':fpr,'positives' : pos}, index=[0])
    return(result)
