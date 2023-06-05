import tensorflow as tf
import sys
sys.path.append('recommenders/models/deeprec/models/sequential/')
from recommenders.models.deeprec.models.sequential.sequential_base_model import (
    SequentialBaseModel,
)
from tensorflow.compat.v1.nn.rnn_cell import RNNCell

__all__ = ["Model"]

def get_shape(inputs):
    dynamic_shape = tf.shape(inputs)
    static_shape = inputs.get_shape().as_list()
    shape = []
    for i, dim in enumerate(static_shape):
        shape.append(dim if dim is not None else dynamic_shape[i])

    return shape


class Model(SequentialBaseModel):
    
    def _build_seq_graph(self):
        
        hparams = self.hparams
        W_initializer = self.initializer
        
        history_emb = tf.concat([self.item_history_embedding, self.cate_history_embedding], -1)
        fatigue_history_emb = tf.concat([self.item_fatigue_history_embedding, self.cate_fatigue_history_embedding], -1)        
        self.mi_extractor = SANetwork(get_shape(history_emb)[-1], get_shape(history_emb)[1], hparams.num_interests, W_initializer)
        if hparams.fatigue_weight:
            self.interest_cross = CrossNetwork(hparams.recent_k, hparams.recent_k//3, hparams.num_cross_layers, hparams.num_dense_layers, initializer=W_initializer, name='interest_cross')
        self.fatigue_cross = CrossNetwork(hparams.num_interests, hparams.num_interests//2, hparams.num_cross_layers, hparams.num_dense_layers, initializer=W_initializer, to_weight=False, name='fatigue_cross')
        if self.hparams.fatigue_emb:
            self.fatigue_tcn = TCN(hparams.recent_k, hparams.num_interests*2, hparams.k_size, 'causal', hparams.conv_channels, W_initializer, name='fatigue_tcn')
        fatigue_cell = tf.keras.layers.StackedRNNCells([FRUCell(rnn_dim, get_shape(history_emb)[-1], W_initializer, W_initializer, name=f'FRUCell_{idx}') for idx, rnn_dim in enumerate(hparams.rnn_dims)])
        self.fatigue_rnn = tf.keras.layers.RNN(fatigue_cell)
        
        interest_output_rec, fatigue_output_rec = self._basic_build_graph(self.user_embedding, history_emb, self.target_item_embedding, self.iterator.mask, self.iterator.CL_mask, self.iterator.recent_idx)
        interest_output_fatigue, fatigue_output_fatigue = self._basic_build_graph(self.users_fatigue_embedding, fatigue_history_emb, self.target_item_fatigue_embedding, self.iterator.fatigue_mask, self.iterator.CL_fatigue_mask, self.iterator.recent_fatigue_idx)
        return interest_output_rec, fatigue_output_rec, interest_output_fatigue, fatigue_output_fatigue
    
    def _basic_build_graph(self, user_emb, history_emb, target_item_embedding, mask, CL_mask, recent_idx):
        interest_emb = self.mi_extractor(history_emb, mask)
        recent_mask = (recent_idx>=0)
        float_recent_mask = tf.cast(recent_mask, tf.float32)
        recent_item_emb = tf.gather(history_emb, recent_idx, axis=1, batch_dims=1)
        interest_norm = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(interest_emb), -1)), 1)
        target_item_emb = tf.expand_dims(target_item_embedding, axis=1)
        M = tf.matmul(recent_item_emb-target_item_emb, interest_emb, transpose_b=True)/(interest_norm+1e-8)
        M = 1 / (1 + tf.math.abs(M))
        M = tf.where(tf.tile(tf.expand_dims(recent_mask, -1), [1, 1, self.hparams.num_interests]), M, tf.zeros_like(M))
        
        if self.hparams.fatigue_weight:
            trans_M = tf.transpose(M, [0, 2, 1])
            interests_fatigue_weight = self.interest_cross(trans_M)
            interests_fatigue_weight = tf.nn.softmax(interests_fatigue_weight*tf.expand_dims(CL_mask, -1), 1)
            fused_interest_emb = tf.reduce_sum(interest_emb*interests_fatigue_weight, 1)
        else:
            fused_interest_emb = tf.reduce_mean(interest_emb, 1)
            
        cross_M = self.fatigue_cross(M)
        recent_fatigue = tf.squeeze(self._fcn_net(cross_M, [get_shape(cross_M)[-1]//2], scope="fatigue_short"), -1)
        if self.hparams.fatigue_emb:
            conv_M = self.fatigue_tcn(cross_M)
            recent_emb_feat = tf.concat([recent_item_emb, recent_item_emb, conv_M*tf.expand_dims(CL_mask, -1)], -1)
            short_interest_emb = self.fatigue_rnn(recent_emb_feat, mask=recent_mask, initial_state=fused_interest_emb)
        else:
            recent_emb_feat = tf.concat([recent_item_emb, recent_item_emb], -1)
            short_interest_emb = self.fatigue_rnn(recent_emb_feat, mask=recent_mask, initial_state=fused_interest_emb)
        interest_output = tf.concat([fused_interest_emb, short_interest_emb, target_item_embedding], -1)
        
        recent_fatigue = tf.reduce_sum(recent_fatigue*float_recent_mask, -1, keepdims=True)/tf.reduce_sum(float_recent_mask, -1, keepdims=True)

        return interest_output, recent_fatigue

class CrossNetwork(tf.keras.layers.Layer):
    def __init__(self, dim=15, proj_dim=5, num_cross_layers=2, num_dense_layers=2, initializer=None, to_weight=True, **kwargs):
        super(CrossNetwork, self).__init__(**kwargs)
        self.W_initializer = initializer
        self.num_cross_layers = num_cross_layers
        self.num_dense_layers = num_dense_layers
        self.proj_dim = proj_dim
        self.dim = dim
        self.to_weight = to_weight
        
        self.cross_U, self.cross_V = [], []
        for idx in range(self.num_cross_layers):
            self.cross_U.append(tf.keras.layers.Dense(self.proj_dim, kernel_initializer=self.W_initializer, use_bias=False, name=f'denseU_{idx}'))
            self.cross_V.append(tf.keras.layers.Dense(self.dim, kernel_initializer=self.W_initializer, bias_initializer=self.W_initializer, name=f'denseV_{idx}'))
        
        self.wide_deep = tf.keras.Sequential()
        for idx in range(self.num_dense_layers):
            self.wide_deep.add(tf.keras.layers.Dense(dim, kernel_initializer=self.W_initializer, bias_initializer=self.W_initializer, activation=tf.nn.leaky_relu, name=f'wide_deep_{idx}'))
        if to_weight:
            self.weight_dense = tf.keras.Sequential()
            self.weight_dense.add(tf.keras.layers.Dense(dim, kernel_initializer=self.W_initializer, bias_initializer=self.W_initializer, activation=tf.nn.leaky_relu, name='fatigue_weight_dense1'))
            self.weight_dense.add(tf.keras.layers.Dense(1, kernel_initializer=self.W_initializer, bias_initializer=self.W_initializer, activation=None, name='fatigue_weight_dense2'))
        
    def call(self, X0):
        with tf.compat.v1.variable_scope('cross', initializer=self.W_initializer):
            Xi = X0
            for denseU, denseV in zip(self.cross_U, self.cross_V):
                Xi = X0 * denseV(denseU(Xi)) + Xi
            deep_X = self.wide_deep(X0)
            output = tf.concat([Xi, deep_X], -1)
            if self.to_weight:
                return self.weight_dense(output)
            else:
                return output

class TCN(tf.keras.layers.Layer):

    def __init__(self, seqlen, feature_dim, k_size, padding, conv_channels, initializer, **kwargs):
        super(TCN, self).__init__(**kwargs)
        
        self.seqlen = seqlen
        self.feature_dim = feature_dim
        self.conv_channels = conv_channels
        self.W_initializer = initializer
        self.k_size = k_size
            
        self.TCN_feature = []
        for d, nc in zip([feature_dim]+self.conv_channels[:-1], self.conv_channels):
            this_TCN_feature = tf.keras.Sequential()
            this_TCN_feature.add(tf.keras.layers.Conv1D(nc, self.k_size, padding=padding, activation=tf.nn.leaky_relu, input_shape=[self.seqlen, d]))
            self.TCN_feature.append(this_TCN_feature)

    def call(self, features):
        with tf.compat.v1.variable_scope('tcn_feature', initializer=self.W_initializer):
            current_features = features
            for this_TCN_feature in self.TCN_feature:
                current_features = this_TCN_feature(current_features)
            return current_features
    
class FRUCell(RNNCell):
    def __init__(self, state_dim, emb_dim, kernel_initializer, bias_initializer, reuse=None, name=None, **kwargs):
        super(FRUCell, self).__init__(_reuse=reuse, name=name, **kwargs)
        self.state_dim = state_dim
        self.emb_dim = emb_dim
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        
    @property
    def state_size(self):
        return self.state_dim

    @property
    def output_size(self):
        return self.state_dim
    
    def build(self, inputs_shape):
        self.feature_dim = inputs_shape[-1] - self.emb_dim
        
        self.W_feat_reset = self.add_weight(shape=(self.feature_dim, self.state_dim),
                                            initializer=self.kernel_initializer,
                                            name='W_feat_reset')
        self.W_feat_update = self.add_weight(shape=(self.feature_dim, self.state_dim),
                                            initializer=self.kernel_initializer,
                                            name='W_feat_update')
        self.W_feat_state_reset = self.add_weight(shape=(self.state_dim, self.state_dim),
                                            initializer=self.kernel_initializer,
                                            name='W_feat_state_reset')
        self.W_feat_state_update = self.add_weight(shape=(self.state_dim, self.state_dim),
                                            initializer=self.kernel_initializer,
                                            name='W_feat_state_update')
        self.b_feat_reset = self.add_weight(shape=(1, self.state_dim),
                                            initializer=self.bias_initializer,
                                            name='b_feat_reset')
        self.b_feat_update = self.add_weight(shape=(1, self.state_dim),
                                            initializer=self.bias_initializer,
                                            name='b_feat_update')
        self.W_emb = self.add_weight(shape=(self.emb_dim, self.state_dim),
                                            initializer=self.kernel_initializer,
                                            name='W_emb')
        self.W_state = self.add_weight(shape=(self.state_dim, self.state_dim),
                                            initializer=self.kernel_initializer,
                                            name='W_state')
        self.b_emb = self.add_weight(shape=(1, self.state_dim),
                                            initializer=self.bias_initializer,
                                            name='b_emb')
        self.built = True
    
    def call(self, inputs, state):
        emb_inputs, feature_inputs = tf.split(inputs, [self.emb_dim, self.feature_dim], -1)
        feat_z = tf.nn.sigmoid(tf.matmul(feature_inputs, self.W_feat_update) +\
                          tf.matmul(state, self.W_feat_state_update) +\
                          self.b_feat_update)
        feat_r = tf.nn.sigmoid(tf.matmul(feature_inputs, self.W_feat_reset) +\
                          tf.matmul(state, self.W_feat_state_reset) +\
                          self.b_feat_reset)
        h_hat = tf.nn.tanh(tf.matmul(emb_inputs, self.W_emb) +\
                        tf.matmul(feat_r*state, self.W_state) +\
                        self.b_emb)
        h = feat_z*state + (1-feat_z)*h_hat
        return h, h

        
class SANetwork(tf.keras.layers.Layer):
    def __init__(self, dim, seq_len, num_interest=4, initializer=None):
        super(SANetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(dim*4, activation=tf.nn.tanh, name='dense1')
        self.dense2 = tf.keras.layers.Dense(num_interest, name='dense2')
        self.seq_len = seq_len
        self.num_interest = num_interest
        self.initializer = initializer
        
    def call(self, item_his_emb, mask):
        with tf.compat.v1.variable_scope('SA', initializer=self.initializer):
            item_att_w = self.dense2(self.dense1(item_his_emb))
        item_att_w  = tf.transpose(item_att_w, [0, 2, 1])

        atten_mask = tf.tile(tf.expand_dims(mask, axis=1), [1, self.num_interest, 1])
        paddings = tf.ones_like(atten_mask, dtype=tf.float32) * (-2 ** 32 + 1)

        item_att_w = tf.where(tf.equal(atten_mask, 0), paddings, item_att_w)
        item_att_w = tf.nn.softmax(item_att_w)

        interest_emb = tf.matmul(item_att_w, item_his_emb)

        return interest_emb