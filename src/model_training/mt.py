import os
from Mylib import (
    myfuncs,
    tf_myfuncs,
    tf_model_training_funcs,
    tf_model_training_classes,
    tf_myclasses,
    tf_model_tuner,
)
import tensorflow as tf
from src.utils import classes
import numpy as np
import time
import gc
import itertools
from sklearn.model_selection import ParameterSampler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Input
import pandas as pd


def load_data(data_transformation_path, class_names_path):
    train_ds = tf.data.Dataset.load(os.path.join(data_transformation_path, "train_ds"))
    val_ds = tf.data.Dataset.load(os.path.join(data_transformation_path, "val_ds"))
    vocab_size = myfuncs.load_python_object(
        f"{data_transformation_path}/vocab_size.pkl"
    )
    num_classes = len(myfuncs.load_python_object(class_names_path))

    return train_ds, val_ds, vocab_size, num_classes


def load_data_for_transformer(data_transformation_path, class_names_path):
    train_ds = tf.data.Dataset.load(os.path.join(data_transformation_path, "train_ds"))
    val_ds = tf.data.Dataset.load(os.path.join(data_transformation_path, "val_ds"))
    vocab_size = myfuncs.load_python_object(
        f"{data_transformation_path}/vocab_size.pkl"
    )
    output_sequence_length = myfuncs.load_python_object(
        f"{data_transformation_path}/output_sequence_length.pkl"
    )
    num_classes = len(myfuncs.load_python_object(class_names_path))

    return train_ds, val_ds, vocab_size, output_sequence_length, num_classes


# DCN
def create_model(param):
    """Tạo model với cấu trúc như sau <br>
    ```
    Input
    DenseBatchNormalizationDropoutTuner
    Dense không có Dropout
    output
    ```
    """
    input_layer = tf.keras.Input(shape=(param["vocab_size"],))
    output_layer = tf_model_training_funcs.get_output_layer(param["num_classes"])
    dense_layer = tf_model_training_classes.LayerCreator(param, "layer0").next()

    x = dense_layer(input_layer)
    x = output_layer(x)

    model = tf.keras.models.Model(inputs=input_layer, outputs=x)
    return model


# RNN
def create_rnn_model(param):
    input_layer = tf.keras.Input(shape=(None,))
    output_layer = tf_model_training_funcs.get_output_layer(param["num_classes"])

    embedding_layer = tf.keras.layers.Embedding(
        input_dim=param["vocab_size"], output_dim=param["output_dim"], mask_zero=True
    )
    bidirectional_lstm_layer = tf_model_training_classes.LayerCreator(
        param, "layer0"
    ).next()
    dense_layer = tf_model_training_classes.LayerCreator(param, "layer1").next()

    x = embedding_layer(input_layer)
    x = bidirectional_lstm_layer(x)
    x = dense_layer(x)
    x = output_layer(x)

    model = tf.keras.models.Model(inputs=input_layer, outputs=x)
    return model


# Transfomer
def create_transformer_model(param):
    input_layer = tf.keras.Input(shape=(None,))
    output_layer = tf_model_training_funcs.get_output_layer(param["num_classes"])
    position_embedding_layer = tf_myclasses.PositionalEmbedding(
        sequence_length=param["output_sequence_length"],
        input_dim=param["vocab_size"],
        output_dim=param["embed_dim"],
    )
    transformer_encoder_layer = tf_model_tuner.TransformerEncoderList(
        embed_dim=param["embed_dim"],
        dense_dim=param["dense_dim"],
        num_heads=param["num_heads"],
        num_layers=param["num_layers"],
    )
    flatten_layer = tf.keras.layers.GlobalMaxPooling1D()
    dense_layer = tf_model_training_classes.LayerCreator(param, "layer0").next()

    x = position_embedding_layer(input_layer)
    x = transformer_encoder_layer(x)
    x = flatten_layer(x)
    x = dense_layer(x)
    x = output_layer(x)

    model = tf.keras.models.Model(inputs=input_layer, outputs=x)
    return model
