# DeepSIBA Implementation in Tensorflow 2.3 (Still Under Construction!!!)

# DeepSIBA: Chemical Structure-based Inference of Biological Alterations
### Christos Fotis<sup>1(+)</sup>, Nikolaos Meimetis<sup>1+</sup>, Antonios Sardis<sup>1</sup>, Leonidas G.Alexopoulos<sup>1,2(*)</sup>
 #### 1. BioSys Lab, National Technical University of Athens, Athens, Greece.
#### 2. ProtATonce Ltd, Athens, Greece.

(+)Equal contributions

(*)Correspondence to: leo@mail.ntua.gr

Github repository of the study:
> DeepSIBA: Chemical Structure-based Inference of Biological Alterations <br>
> C.Fotis<sup>1(+)</sup>, N.Meimetis<sup>1+</sup>, A.Sardis<sup>1</sup>, LG. Alexopoulos<sup>1,2(*)</sup>

# DeepSIBA Approach
![figure1_fl_02](https://user-images.githubusercontent.com/48244638/80740035-212c7f00-8b20-11ea-9d97-300758595403.png)

## Clone
```bash
# clone the source code on your directory
$ git clone https://github.com/NickMeim/deepSIBA_tf2.git
```
# Learning directory overview

This directory contains the required data and source code to implement the machine learning part of deepSIBA. 

- The data, trained_models and screening folders contain a readme with the appropriate download instructions and explanations.

The NGF and NGF layers folders contain the source code to implement the graph convolution layers and the appropriate featurization. The code was adapted to Keras from https://github.com/keiserlab/keras-neural-graph-fingerprint.

The utility folder contains the following functions:

- A Keras training generator and a predicting generator
- A function to evaluate the performance of deepSIBA
- Custom layer and loss function to implement the Gaussian regression layer.

**The notebook deepSIBA_examples describes how to implement and utilize deepSIBA.**

The main functions that implement deepSIBA are:

1. deepSIBA_model.py
2. deepSIBA_train.py
3. deepSIBA_ensembles.py
4. deepSIBA_screening.py

The input required for each function is described thoroughly in the deepSIBA_examples notebook.
