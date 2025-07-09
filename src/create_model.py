from tensorflow.keras import layers, models
from src import create_object
from src.layers import PositionalEmbedding


def create_model(param):
    inputs = layers.Input(shape=(param["max_tokens"],))

    dense_layer = create_object.create_object(param, "dense")
    output_layer = layers.Dense(units=2, activation="softmax")

    x = dense_layer(inputs)
    x = output_layer(x)

    model = models.Model(inputs, x)
    return model


def create_model_for_rnn(param):
    inputs = layers.Input(shape=(None,), dtype="int64")
    embedding_layer = layers.Embedding(
        input_dim=param["max_tokens"], output_dim=param["embedding_output_dim"]
    )
    bidirectional_layer = create_object.create_object(param, "bidirectional_rnn")
    dense_layer = create_object.create_object(param, "dense")
    output_layer = layers.Dense(units=2, activation="softmax")

    x = embedding_layer(inputs)
    x = bidirectional_layer(x)
    x = dense_layer(x)
    x = output_layer(x)

    model = models.Model(inputs, x)
    return model


def create_model_for_transformer(param):
    print("create_model_for_transformer")

    inputs = layers.Input(shape=(None,), dtype="int64")
    positional_embedding_layer = PositionalEmbedding(
        sequence_length=param["output_sequence_length"],
        input_dim=param["max_tokens"],
        output_dim=param["posititional_embedding_output_dim"],
    )
    transformer_layer = create_object.create_object(param, "transformer")
    transformer_layer.update_embed_dim(param["posititional_embedding_output_dim"])
    flatten_layer = create_object.create_object(param, "flatten")
    dense_layer = create_object.create_object(param, "dense")
    output_layer = layers.Dense(units=2, activation="softmax")

    x = positional_embedding_layer(inputs)
    x = transformer_layer(x)
    x = flatten_layer(x)
    x = dense_layer(x)
    x = output_layer(x)

    model = models.Model(inputs, x)
    return model
