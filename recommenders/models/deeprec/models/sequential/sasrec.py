import tensorflow as tf
from recommenders.models.deeprec.models.sequential.sequential_base_model import (
    SequentialBaseModel,
)

__all__ = ["SASRec"]

def get_shape(inputs):
    dynamic_shape = tf.shape(inputs)
    static_shape = inputs.get_shape().as_list()
    shape = []
    for i, dim in enumerate(static_shape):
        shape.append(dim if dim is not None else dynamic_shape[i])

    return shape


class SASRec(SequentialBaseModel):
    def _build_seq_graph(self):
        seq_embeddings = tf.concat([self.item_history_embedding, self.cate_history_embedding], -1)
        _, seq_max_len, embedding_dim = get_shape(seq_embeddings)
        self.encoder = Encoder(
            self.hparams.num_blocks,
            seq_max_len,
            embedding_dim,
            embedding_dim,
            self.hparams.attention_num_heads,
            [embedding_dim, embedding_dim],
            0.5,
        )
        self.layer_normalization = LayerNormalization(
            1, embedding_dim, 1e-08
        )
        
        seq_embeddings = seq_embeddings * (embedding_dim ** 0.5)
        positional_seq = tf.expand_dims(tf.range(seq_max_len), 0)
        positional_embeddings = tf.compat.v1.nn.embedding_lookup(params=self.pos_lookup, ids=positional_seq)
        seq_embeddings += positional_embeddings
        
        self.dropout_layer = tf.keras.layers.Dropout(0.5)
        seq_embeddings = self.dropout_layer(seq_embeddings)
        seq_embeddings *= tf.expand_dims(tf.cast(self.iterator.mask, tf.float32), -1)
        
        seq_attention = seq_embeddings
        # seq_attention = self.encoder(seq_attention, self.is_train_stage, tf.cast(self.iterator.mask, tf.int32))
        seq_attention = self.encoder(seq_attention, self.is_train_stage)
        seq_attention = self.layer_normalization(seq_attention)
        # model_output = tf.squeeze(seq_attention, 1)
        model_output = tf.squeeze(tf.gather(seq_attention, tf.reduce_sum(tf.cast(self.iterator.mask, tf.int32), -1, keepdims=True)-1, axis=1, batch_dims=1), 1)
        # model_output = tf.concat([model_output, self.target_item_embedding], -1)
        return model_output

class MultiHeadAttention(tf.keras.layers.Layer):
    """
    - Q (query), K (key) and V (value) are split into multiple heads (num_heads)
    - each tuple (q, k, v) are fed to scaled_dot_product_attention
    - all attention outputs are concatenated
    """

    def __init__(self, attention_dim, num_heads, dropout_rate):
        """Initialize parameters.

        Args:
            attention_dim (int): Dimension of the attention embeddings.
            num_heads (int): Number of heads in the multi-head self-attention module.
            dropout_rate (float): Dropout probability.
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        assert attention_dim % self.num_heads == 0
        self.dropout_rate = dropout_rate

        self.depth = attention_dim // self.num_heads

        self.Q = tf.keras.layers.Dense(self.attention_dim, activation=None)
        self.K = tf.keras.layers.Dense(self.attention_dim, activation=None)
        self.V = tf.keras.layers.Dense(self.attention_dim, activation=None)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, queries, keys):
        """Model forward pass.

        Args:
            queries (tf.Tensor): Tensor of queries.
            keys (tf.Tensor): Tensor of keys

        Returns:
            tf.Tensor: Output tensor.
        """
        # Linear projections
        Q = self.Q(queries)  # (N, T_q, C)
        K = self.K(keys)  # (N, T_k, C)
        V = self.V(keys)  # (N, T_k, C)

        # --- MULTI HEAD ---
        # Split and concat, Q_, K_ and V_ are all (h*N, T_q, C/h)
        Q_ = tf.concat(tf.split(Q, self.num_heads, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, self.num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, self.num_heads, axis=2), axis=0)

        # --- SCALED DOT PRODUCT ---
        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [self.num_heads, 1])  # (h*N, T_k)
        # key_masks = tf.expand_dims(key_masks, 1)
        key_masks = tf.tile(
            tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]
        )  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-(2 ** 32) + 1)
        # outputs, (h*N, T_q, T_k)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)

        # Future blinding (Causality)
        diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
        tril = tf.linalg.LinearOperatorLowerTriangular(
            diag_vals
        ).to_dense()  # (T_q, T_k)
        masks = tf.tile(
            tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]
        )  # (h*N, T_q, T_k)

        paddings = tf.ones_like(masks) * (-(2 ** 32) + 1)
        # outputs, (h*N, T_q, T_k)
        outputs = tf.where(tf.equal(masks, 0), paddings, outputs)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking, query_masks (N, T_q)
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))
        query_masks = tf.tile(query_masks, [self.num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(
            tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]
        )  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = self.dropout(outputs)

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # --- MULTI HEAD ---
        # concat heads
        outputs = tf.concat(
            tf.split(outputs, self.num_heads, axis=0), axis=2
        )  # (N, T_q, C)

        # Residual connection
        outputs += queries

        return outputs


class PointWiseFeedForward(tf.keras.layers.Layer):
    """
    Convolution layers with residual connection
    """

    def __init__(self, conv_dims, dropout_rate):
        """Initialize parameters.

        Args:
            conv_dims (list): List of the dimensions of the Feedforward layer.
            dropout_rate (float): Dropout probability.
        """
        super(PointWiseFeedForward, self).__init__()
        self.conv_dims = conv_dims
        self.dropout_rate = dropout_rate
        self.conv_layer1 = tf.keras.layers.Conv1D(
            filters=self.conv_dims[0], kernel_size=1, activation="relu", use_bias=True
        )
        self.conv_layer2 = tf.keras.layers.Conv1D(
            filters=self.conv_dims[1], kernel_size=1, activation=None, use_bias=True
        )
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, x):
        """Model forward pass.

        Args:
            x (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Output tensor.
        """

        output = self.conv_layer1(x)
        output = self.dropout_layer(output)

        output = self.conv_layer2(output)
        output = self.dropout_layer(output)

        # Residual connection
        output += x

        return output


class EncoderLayer(tf.keras.layers.Layer):
    """
    Transformer based encoder layer

    """

    def __init__(
        self,
        seq_max_len,
        embedding_dim,
        attention_dim,
        num_heads,
        conv_dims,
        dropout_rate,
    ):
        """Initialize parameters.

        Args:
            seq_max_len (int): Maximum sequence length.
            embedding_dim (int): Embedding dimension.
            attention_dim (int): Dimension of the attention embeddings.
            num_heads (int): Number of heads in the multi-head self-attention module.
            conv_dims (list): List of the dimensions of the Feedforward layer.
            dropout_rate (float): Dropout probability.
        """
        super(EncoderLayer, self).__init__()

        self.seq_max_len = seq_max_len
        self.embedding_dim = embedding_dim

        self.mha = MultiHeadAttention(attention_dim, num_heads, dropout_rate)
        self.ffn = PointWiseFeedForward(conv_dims, dropout_rate)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

        self.layer_normalization = LayerNormalization(
            self.seq_max_len, self.embedding_dim, 1e-08
        )

    # def call_(self, x, training, mask):
    #     """Model forward pass.

    #     Args:
    #         x (tf.Tensor): Input tensor.
    #         training (tf.Tensor): Training tensor.
    #         mask (tf.Tensor): Mask tensor.

    #     Returns:
    #         tf.Tensor: Output tensor.
    #     """

    #     attn_output = self.mha(queries=self.layer_normalization(x), keys=x)
    #     attn_output = self.dropout1(attn_output, training=training)
    #     out1 = self.layernorm1(x + attn_output)

    #     # feed forward network
    #     ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    #     ffn_output = self.dropout2(ffn_output, training=training)
    #     out2 = self.layernorm2(
    #         out1 + ffn_output
    #     )  # (batch_size, input_seq_len, d_model)

    #     # masking
    #     out2 *= mask

    #     return out2

    def call(self, x, training):
        """Model forward pass.

        Args:
            x (tf.Tensor): Input tensor.
            training (tf.Tensor): Training tensor.
            mask (tf.Tensor): Mask tensor.

        Returns:
            tf.Tensor: Output tensor.
        """

        x_norm = self.layer_normalization(x)
        # attn_output = self.mha(queries=x_norm, keys=x)
        # last_x = tf.gather(x_norm, tf.reduce_sum(mask, -1, keepdims=True)-1, axis=1, batch_dims=1)
        # print(last_x.shape)
        attn_output = self.mha(x_norm, x)
        # print('attn_output1', attn_output.shape)
        attn_output = self.ffn(attn_output)
        # print('attn_output2', attn_output.shape)
        # out = attn_output * mask

        return attn_output


class Encoder(tf.keras.layers.Layer):
    """
    Invokes Transformer based encoder with user defined number of layers

    """

    def __init__(
        self,
        num_layers,
        seq_max_len,
        embedding_dim,
        attention_dim,
        num_heads,
        conv_dims,
        dropout_rate,
    ):
        """Initialize parameters.

        Args:
            num_layers (int): Number of layers.
            seq_max_len (int): Maximum sequence length.
            embedding_dim (int): Embedding dimension.
            attention_dim (int): Dimension of the attention embeddings.
            num_heads (int): Number of heads in the multi-head self-attention module.
            conv_dims (list): List of the dimensions of the Feedforward layer.
            dropout_rate (float): Dropout probability.
        """
        super(Encoder, self).__init__()

        self.num_layers = num_layers

        self.enc_layers = [
            EncoderLayer(
                seq_max_len,
                embedding_dim,
                attention_dim,
                num_heads,
                conv_dims,
                dropout_rate,
            )
            for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training):
        """Model forward pass.

        Args:
            x (tf.Tensor): Input tensor.
            training (tf.Tensor): Training tensor.
            mask (tf.Tensor): Mask tensor.

        Returns:
            tf.Tensor: Output tensor.
        """

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training)

        return x  # (batch_size, input_seq_len, d_model)


class LayerNormalization(tf.keras.layers.Layer):
    """
    Layer normalization using mean and variance
    gamma and beta are the learnable parameters
    """

    def __init__(self, seq_max_len, embedding_dim, epsilon):
        """Initialize parameters.

        Args:
            seq_max_len (int): Maximum sequence length.
            embedding_dim (int): Embedding dimension.
            epsilon (float): Epsilon value.
        """
        super(LayerNormalization, self).__init__()
        self.seq_max_len = seq_max_len
        self.embedding_dim = embedding_dim
        self.epsilon = epsilon
        self.params_shape = (self.seq_max_len, self.embedding_dim)
        g_init = tf.ones_initializer()
        self.gamma = tf.Variable(
            initial_value=g_init(shape=self.params_shape, dtype="float32"),
            trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.beta = tf.Variable(
            initial_value=b_init(shape=self.params_shape, dtype="float32"),
            trainable=True,
        )

    def call(self, x):
        """Model forward pass.

        Args:
            x (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Output tensor.
        """
        mean, variance = tf.nn.moments(x, [-1], keepdims=True)
        normalized = (x - mean) / ((variance + self.epsilon) ** 0.5)
        output = self.gamma * normalized + self.beta
        return output