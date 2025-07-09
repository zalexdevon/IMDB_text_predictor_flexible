from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM, GRU
import tensorflow as tf


class DenseLayer(layers.Layer):
    def __init__(self, units, dropout_rate=0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config

    def build(self, input_shape):
        self.Dense = layers.Dense(units=self.units, use_bias=False)
        self.BatchNormalization = layers.BatchNormalization()
        self.Activation = layers.Activation("relu")
        self.Dropout = layers.Dropout(self.dropout_rate)

        super().build(input_shape)

    def call(self, x):
        x = self.Dense(x)
        x = self.BatchNormalization(x)
        x = self.Activation(x)
        x = self.Dropout(x)

        return x

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class PassThroughLayer(layers.Layer):
    """Đơn giản là placeholdout layer, không biến đổi gì cả"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):

        super().build(input_shape)

    def call(self, x):

        return x

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class DenseLayerList(layers.Layer):
    def __init__(self, dropout_rate, list_units, do_have_last_layer, **kwargs):
        super().__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.list_units = list_units
        self.do_have_last_layer = do_have_last_layer

    def get_config(self):
        # Trả về cấu hình của lớp tùy chỉnh
        config = super().get_config()
        config.update(
            {
                "dropout_rate": self.dropout_rate,
                "list_units": self.list_units,
                "do_have_last_layer": self.do_have_last_layer,
            }
        )
        return config

    def build(self, input_shape):
        self.DenseLayers = [
            DenseLayer(units=units, dropout_rate=self.dropout_rate)
            for units in self.list_units
        ]

        self.lastDenseLayer = (
            DenseLayer(units=self.list_units[-1], dropout_rate=0)
            if self.do_have_last_layer
            else PassThroughLayer()
        )

        super().build(input_shape)

    def call(self, x):
        # Xử lí x
        for layer in self.DenseLayers:
            x = layer(x)

        x = self.lastDenseLayer(x)

        return x

    @classmethod
    def from_config(cls, config):
        # Giải mã lại lớp từ cấu hình
        return cls(**config)


class DenseLayerList1(layers.Layer):
    def __init__(
        self,
        type,
        list_list_units,
        list_dropout_rate,
        list_do_have_last_layer,
        single_units,
        single_dropout_rate,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.type = type
        self.list_list_units = list_list_units
        self.list_dropout_rate = list_dropout_rate
        self.list_do_have_last_layer = list_do_have_last_layer
        self.single_units = single_units
        self.single_dropout_rate = single_dropout_rate

    def get_config(self):
        # Trả về cấu hình của lớp tùy chỉnh
        config = super().get_config()
        config.update(
            {
                "type": self.type,
                "list_list_units": self.list_list_units,
                "list_dropout_rate": self.list_dropout_rate,
                "list_do_have_last_layer": self.list_do_have_last_layer,
                "single_units": self.single_units,
                "single_dropout_rate": self.single_dropout_rate,
            }
        )
        return config

    def build(self, input_shape):
        if self.type == "list":
            self.DenseLayers = [
                DenseLayer(units=units, dropout_rate=self.list_dropout_rate)
                for units in self.list_list_units
            ]

            self.lastDenseLayer = (
                DenseLayer(units=self.list_list_units[-1], dropout_rate=0)
                if self.list_do_have_last_layer
                else PassThroughLayer()
            )
        elif self.type == "single":
            self.DenseLayer = DenseLayer(
                units=self.single_units, dropout_rate=self.single_dropout_rate
            )

        super().build(input_shape)

    def call(self, x):
        if self.type == "list":
            for layer in self.DenseLayers:
                x = layer(x)

            x = self.lastDenseLayer(x)
        elif self.type == "single":
            x = self.DenseLayer(x)

        return x

    @classmethod
    def from_config(cls, config):
        # Giải mã lại lớp từ cấu hình
        return cls(**config)


class BidirectionalRNNLayer(layers.Layer):
    def __init__(
        self,
        layer_name,
        units,
        return_sequences=False,
        recurrent_dropout=0,
        merge_mode="concat",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.layer_name = layer_name
        self.units = units
        self.return_sequences = return_sequences
        self.recurrent_dropout = recurrent_dropout
        self.merge_mode = merge_mode

    def get_config(self):
        # Trả về cấu hình của lớp tùy chỉnh
        config = super().get_config()
        config.update(
            {
                "layer_name": self.layer_name,
                "units": self.units,
                "return_sequences": self.return_sequences,
                "recurrent_dropout": self.recurrent_dropout,
                "merge_mode": self.merge_mode,
            }
        )
        return config

    def build(self, input_shape):
        ClassName = globals()[self.layer_name]

        self.BidirectionalRNN = layers.Bidirectional(
            ClassName(
                units=self.units,
                return_sequences=self.return_sequences,
                recurrent_dropout=self.recurrent_dropout,
            ),
            merge_mode=self.merge_mode,
        )
        self.LayerNormalization = layers.LayerNormalization()

        super().build(input_shape)

    def call(self, x):
        x = self.BidirectionalRNN(x)
        x = self.LayerNormalization(x)

        return x

    @classmethod
    def from_config(cls, config):
        # Giải mã lại lớp từ cấu hình
        return cls(**config)


class BidirectionalRNNLayerList(layers.Layer):
    def __init__(
        self,
        layer_name,
        list_units,
        recurrent_dropout=0,
        merge_mode="concat",
        do_have_last_layer=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.layer_name = layer_name
        self.list_units = list_units
        self.recurrent_dropout = recurrent_dropout
        self.merge_mode = merge_mode
        self.do_have_last_layer = do_have_last_layer

    def get_config(self):
        # Trả về cấu hình của lớp tùy chỉnh
        config = super().get_config()
        config.update(
            {
                "layer_name": self.layer_name,
                "list_units": self.list_units,
                "recurrent_dropout": self.recurrent_dropout,
                "merge_mode": self.merge_mode,
                "do_have_last_layer": self.do_have_last_layer,
            }
        )
        return config

    def build(self, input_shape):
        self.BidirectionalRNNLayerList = [
            BidirectionalRNNLayer(
                layer_name=self.layer_name,
                units=units,
                return_sequences=True,
                recurrent_dropout=self.recurrent_dropout,
                merge_mode=self.merge_mode,
            )
            for units in self.list_units
        ]

        self.lastBidirectionalRNNLayer = [
            (
                BidirectionalRNNLayer(
                    layer_name=self.layer_name,
                    units=self.list_units[-1],
                    return_sequences=False,
                    recurrent_dropout=self.recurrent_dropout,
                    merge_mode=self.merge_mode,
                )
                if self.do_have_last_layer
                else PassThroughLayer()
            )
        ]

        super().build(input_shape)

    def call(self, x):
        for layer in self.BidirectionalRNNLayerList:
            x = layer(x)

        x = self.lastBidirectionalRNNLayer(x)

        return x

    @classmethod
    def from_config(cls, config):
        # Giải mã lại lớp từ cấu hình
        return cls(**config)


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        # Các siêu tham số
        self.supports_masking = True  # Có hỗ trợ masking
        self.embed_dim = embed_dim  # size của input token vectors
        self.dense_dim = dense_dim  # Size của Denser layer
        self.num_heads = num_heads  # Số lượng attention heads

        # Các layers
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = tf.keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):  # Tính toán là ở trong hàm call()
        if mask is not None:
            mask = mask[
                :, tf.newaxis, :
            ]  # mask được tạo ra bởi Embedding layer là 2D, nhưng attention layer thì yêu cầu 3D hoặc 4D -> thêm chiều
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def build(self, input_shape):
        super().build(input_shape)

    def get_config(self):  # Cần thiết để lưu model
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "dense_dim": self.dense_dim,
            }
        )
        return config


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):

        super().__init__(**kwargs)
        # Các layers
        self.token_embeddings = layers.Embedding(
            input_dim=input_dim, output_dim=output_dim
        )  # Embedding layers cho token indices
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )  # Layer này cho token positions

        # Các siêu tham số
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):

        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        embedded = embedded_tokens + embedded_positions  # Cộng 2 embedding vectors lại

        # Save mask using Keras's _keras_mask mechanism
        embedded._keras_mask = tf.not_equal(inputs, 0)
        return embedded

    def build(self, input_shape):
        super().build(input_shape)

    def compute_mask(
        self, inputs, mask=None
    ):  # Giống với Embedding layer,  layer này nên tạo ra mask để ignore paddings 0 trong inputs
        return None

    def get_config(self):  # Để lưu được model
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "sequence_length": self.sequence_length,
                "input_dim": self.input_dim,
            }
        )
        return config


class TransformerEncoderList(layers.Layer):
    def __init__(self, num_transformer, dense_dim, num_heads, embed_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.num_transformer = num_transformer
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "embed_dim": self.embed_dim,
                "dense_dim": self.dense_dim,
                "num_heads": self.num_heads,
            }
        )
        return config

    def build(self, input_shape):
        self.TransformerEncoderList = [
            TransformerEncoder(
                embed_dim=self.embed_dim,
                dense_dim=self.dense_dim,
                num_heads=self.num_heads,
            )
            for _ in range(self.num_transformer)
        ]

        super().build(input_shape)

    def call(self, x):
        for layer in self.TransformerEncoderList:
            x = layer(x)

        return x

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def update_embed_dim(self, embed_dim):
        for layer in self.TransformerEncoderList:
            layer.embed_dim = embed_dim
