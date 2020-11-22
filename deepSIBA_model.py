from __future__ import division, print_function
import numpy as np
from numpy import inf, ndarray
import pandas as pd
import tensorflow as tf
import os
import random
import keras
import sklearn
import re
from keras import optimizers, losses, regularizers
import keras.backend as K
from keras.models import model_from_json, load_model, Model
from tempfile import TemporaryFile
from keras import layers
from keras.callbacks import History, ReduceLROnPlateau
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, Layer
from keras.initializers import glorot_normal
from keras.regularizers import l2
from functools import partial
from multiprocessing import cpu_count, Pool
from keras.utils.generic_utils import Progbar
from copy import deepcopy
from NGF.utils import filter_func_args, mol_shapes_to_dims
import NGF.utils
import NGF_layers.features
import NGF_layers.graph_layers
from NGF_layers.features import one_of_k_encoding, one_of_k_encoding_unk, atom_features, bond_features, num_atom_features, num_bond_features, padaxis, tensorise_smiles, concat_mol_tensors
from NGF_layers.graph_layers import temporal_padding, neighbour_lookup, NeuralGraphHidden, NeuralGraphOutput
from math import ceil
from sklearn.metrics import mean_squared_error
from utility.gaussian import GaussianLayer, custom_loss, ConGaussianLayer
from utility.evaluator import r_square, get_cindex, pearson_r,custom_mse, mse_sliced, model_evaluate

#Define siamese encoder
class enc_graph(keras.Model):
    def __init__(self,params):
        super(enc_graph, self).__init__()

        ### encode smiles
        #atoms0 = tf.keras.layers.InputLayer(name='atom_inputs', input_shape=(params["max_atoms"], params["num_atom_features"],),dtype = 'float32')
        #bonds = tf.keras.layers.InputLayer(name='bond_inputs', input_shape=(params["max_atoms"], params["max_degree"], params["num_bond_features"],),dtype = 'float32')
        #edges = tf.keras.layers.InputLayer(name='edge_inputs', input_shape=(params["max_atoms"], params["max_degree"],), dtype='int32')

        self.g1 = NeuralGraphHidden(params["graph_conv_width"][0] , activ = None, bias = True , init = 'glorot_normal')
        self.bn1 = BatchNormalization(momentum=0.6)
        self.act1 = Activation('relu')

        self.g2 = NeuralGraphHidden(params["graph_conv_width"][1] , activ = None, bias = True , init = 'glorot_normal')
        self.bn2 = BatchNormalization(momentum=0.6)
        self.act2 = Activation('relu')

        self.g3 = NeuralGraphHidden(params["graph_conv_width"][2] , activ = None, bias = True , init = 'glorot_normal')
        self.bn3 = BatchNormalization(momentum=0.6)
        self.act3 = Activation('relu')


        self.conv1d=keras.layers.Conv1D(params["conv1d_filters"], params["conv1d_size"], activation=None, use_bias=False, kernel_initializer='glorot_uniform')
        self.bn4= BatchNormalization(momentum=0.6)
        self.act4 = Activation('relu')
        self.dropout=keras.layers.Dropout(params["dropout_encoder"])

    def call(self,atoms,bonds,edges):
        x1 = self.g1([atoms,bonds,edges])
        x1 = self.bn1(x1)
        x1 = self.act1(x1)

        x2 = self.g2([x1,bonds,edges])
        x2 = self.bn2(x2)
        x2 = self.act2(x2)

        x3 = self.g3([x2,bonds,edges])
        x3 = self.bn3(x3)
        x3 = self.act3(x3)

        x4 = self.conv1d(x3)
        x4 = self.bn4(x4)
        x4 = self.act4(x4)
        x4 = self.dropout(x4)

        return x4


    #End of encoding
    #graph_encoder = keras.Model(inputs=[atoms0, bonds, edges], outputs= g4)

    #print(graph_encoder.summary())
    #return graph_encoder

#Define operations of distance module after the siamese encoders
class siamese_model(keras.Model):
    def __init__(self,params):
        super(siamese_model, self).__init__()

        self.encoder = enc_graph(params)
        self.dist = keras.layers.Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
        self.conv1 = keras.layers.Conv1D(params["conv1d_filters_dist"][0], params["conv1d_size_dist"][0], activation=None, use_bias=False, kernel_initializer='glorot_uniform')
        self.bn1 = BatchNormalization(momentum=0.6)
        self.act1 = Activation('relu')
        self.drop_dist1 = keras.layers.Dropout(params["dropout_dist"])
        self.conv2 = keras.layers.Conv1D(params["conv1d_filters_dist"][1], params["conv1d_size_dist"][1], activation=None, use_bias=False, kernel_initializer='glorot_uniform')
        self.bn2 = BatchNormalization(momentum=0.6)
        self.act2 = Activation('relu')
        self.drop_dist2 = keras.layers.Dropout(params["dropout_dist"])
        self.pool = keras.layers.MaxPooling1D(pool_size= params["pool_size"], strides=None, padding='valid', data_format='channels_last')
        self.bn3 = BatchNormalization(momentum=0.6)
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(params["dense_size"][0],activation = None,kernel_regularizer=regularizers.l2(params["l2reg"]), kernel_initializer='glorot_normal')
        self.bn4 = BatchNormalization(momentum=0.6)
        self.act4 = Activation('relu')
        self.drop_dist4 = keras.layers.Dropout(params["dropout_dist"])
        self.dense2 = keras.layers.Dense(params["dense_size"][1],activation = None,kernel_regularizer=regularizers.l2(params["l2reg"]), kernel_initializer='glorot_normal')
        self.bn5 = BatchNormalization(momentum=0.6)
        self.act5 = Activation('relu')
        self.drop_dist5 = keras.layers.Dropout(params["dropout_dist"])
        self.dense3 = keras.layers.Dense(params["dense_size"][2],activation = None,kernel_regularizer=regularizers.l2(params["l2reg"]), kernel_initializer='glorot_normal')
        self.bn6 = BatchNormalization(momentum=0.6)
        self.act6 = Activation('relu')
        self.drop_dist6 = keras.layers.Dropout(params["dropout_dist"])

    #atoms0_1 = tf.keras.layers.InputLayer(name='atom_inputs_1', input_shape=(params["max_atoms"], params["num_atom_features"],),dtype = 'float32')
    ##bonds_1 = tf.keras.layers.InputLayer(name='bond_inputs_1', input_shape=(params["max_atoms"], params["max_degree"], params["num_bond_features"],),dtype = 'float32')
    #edges_1 = tf.keras.layers.InputLayer(name='edge_inputs_1', input_shape=(params["max_atoms"], params["max_degree"],), dtype='int32')

    ##atoms0_2 = tf.keras.layers.InputLayer(name='atom_inputs_2', input_shape=(params["max_atoms"], params["num_atom_features"],),dtype = 'float32')
    #bonds_2 = tf.keras.layers.InputLayer(name='bond_inputs_2', input_shape=(params["max_atoms"], params["max_degree"], params["num_bond_features"],),dtype = 'float32')
    #edges_2 = tf.keras.layers.InputLayer(name='edge_inputs_2', input_shape=(params["max_atoms"], params["max_degree"],), dtype='int32')

    #graph_encoder = enc_graph(params)

    def call(self,atoms0_1,bonds_1,edges_1,atoms0_2,bonds_2,edges_2):

        encoded_1 = self.encoder(atoms0_1,bonds_1,edges_1)
        encoded_2 = self.encoder(atoms0_2,bonds_2,edges_2)

        #L1_layer = keras.layers.Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
        L1_distance = self.dist([encoded_1, encoded_2])

        x = self.conv1(L1_distance)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.drop_dist1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.drop_dist2(x)

        x = self.pool(x)
        x = self.bn3(x)
        x = self.flatten(x)

        x = self.dense1(x)
        x = self.bn4(x)
        x = self.act4(x)
        x = self.drop_dist4(x)


        x = self.dense2(x)
        x = self.bn5(x)
        x = self.act5(x)
        x = self.drop_dist5(x)

        x = self.dense3(x)
        x = self.bn6(x)
        x = self.act6(x)
        x = self.drop_dist6(x)

        #Final Gaussian Layer to predict mean distance and standard deaviation of distance
        ##if params["ConGauss"]:
        #    mu, sigma = ConGaussianLayer(1, name='main_output')(fc3)
        ##else:
        #    mu, sigma = GaussianLayer(1, name='main_output')(fc3) #default used most of the times
        #siamese_net = Model(inputs = [atoms0_1, bonds_1, edges_1, atoms0_2, bonds_2, edges_2], outputs = mu)

        #thresh = params["dist_thresh"] #threshold to consider similars
        #adam = keras.optimizers.Adam(lr = params["lr"], beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)
        #siamese_net.compile(optimizer = adam,loss= custom_loss(sigma),metrics=['mse', get_cindex, r_square, pearson_r, mse_sliced(thresh)])

        #mu,sigma = self.gaussian(x)

        return x


    #int_net = keras.Model(inputs=[atoms0_1,bonds_1,edges_1,atoms0_2,bonds_2,edges_2],outputs=fc6)
    #print(int_net.summary())
    #return siamese_net

def deepsiba_model(params):

    atoms0_1 = Input(name='atom_inputs_1', shape=(params["max_atoms"], params["num_atom_features"]),dtype = 'float32')
    bonds_1 = Input(name='bond_inputs_1', shape=(params["max_atoms"], params["max_degree"], params["num_bond_features"]),dtype = 'float32')
    edges_1 = Input(name='edge_inputs_1', shape=(params["max_atoms"], params["max_degree"]), dtype='int32')

    atoms0_2 = Input(name='atom_inputs_2',shape=(params["max_atoms"], params["num_atom_features"]),dtype = 'float32')
    bonds_2 = Input(name='bond_inputs_2',shape=(params["max_atoms"], params["max_degree"], params["num_bond_features"]),dtype = 'float32')
    edges_2 = Input(name='edge_inputs_2', shape=(params["max_atoms"], params["max_degree"]), dtype='int32')

    emb=siamese_model(params)(atoms0_1,bonds_1,edges_1,atoms0_2,bonds_2,edges_2)

    #Final Gaussian Layer to predict mean distance and standard deaviation of distance
    if params["ConGauss"]:
        mu, sigma = ConGaussianLayer(1, name='main_output')(emb)
    else:
        mu, sigma = GaussianLayer(1, name='main_output')(emb) #default used most of the time

    out= keras.layers.Concatenate(axis=1)([mu, sigma])
    siamese_net = keras.Model(inputs = [atoms0_1, bonds_1, edges_1, atoms0_2, bonds_2, edges_2], outputs = out)

    thresh = params["dist_thresh"] #threshold to consider similars
    adam = keras.optimizers.Adam(lr = params["lr"], beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)
    siamese_net.compile(optimizer = adam,loss= custom_loss,metrics=[custom_mse, get_cindex, r_square, pearson_r, mse_sliced(thresh)])


    #int_net = keras.Model(inputs=[atoms0_1,bonds_1,edges_1,atoms0_2,bonds_2,edges_2],outputs=fc6)
    #print(int_net.summary())
    return siamese_net
