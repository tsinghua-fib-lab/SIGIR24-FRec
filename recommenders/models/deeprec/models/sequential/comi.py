import tensorflow as tf
from recommenders.models.deeprec.models.sequential.sequential_base_model import (
    SequentialBaseModel,
)

__all__ = ["Comi"]

def get_shape(inputs):
    dynamic_shape = tf.shape(inputs)
    static_shape = inputs.get_shape().as_list()
    shape = []
    for i, dim in enumerate(static_shape):
        shape.append(dim if dim is not None else dynamic_shape[i])

    return shape


class Comi(SequentialBaseModel):
    
    def _build_seq_graph(self):
        
        hparams = self.hparams
        with tf.compat.v1.variable_scope("Comi", initializer=self.initializer):
            history_emb = tf.concat(
                [self.item_history_embedding, self.cate_history_embedding], 2
            )
            if hparams.extractor == 'caps':
                _, model_output = CapsuleNetwork(history_emb.shape.as_list()[-1], history_emb.shape.as_list()[1], initializer=self.initializer, num_interest=hparams.slots)(history_emb, self.target_item_embedding, self.iterator.mask)
            else:
                model_output = SANetwork(history_emb.shape.as_list()[-1], history_emb.shape.as_list()[1], initializer=self.initializer, num_interest=hparams.slots)(history_emb, self.target_item_embedding, self.iterator.mask)
        model_output = tf.concat([model_output, self.target_item_embedding], -1)  
        return model_output
    
class SANetwork(tf.keras.layers.Layer):
    def __init__(self, dim, seq_len, num_interest=4, initializer=None):
        super(SANetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(dim*4, activation=tf.nn.tanh, name='dense1')
        self.dense2 = tf.keras.layers.Dense(num_interest, name='dense2')
        self.seq_len = seq_len
        self.dim = dim
        self.num_interest = num_interest
        self.initializer = initializer
        
    def call(self, item_his_emb, item_eb, mask):
        with tf.compat.v1.variable_scope('SA', initializer=self.initializer):
            item_att_w = self.dense2(self.dense1(item_his_emb))
        item_att_w  = tf.transpose(item_att_w, [0, 2, 1])

        atten_mask = tf.tile(tf.expand_dims(mask, axis=1), [1, self.num_interest, 1])
        paddings = tf.ones_like(atten_mask, dtype=tf.float32) * (-2 ** 32 + 1)

        item_att_w = tf.where(tf.equal(atten_mask, 0), paddings, item_att_w)
        item_att_w = tf.nn.softmax(item_att_w)

        interest_emb = tf.matmul(item_att_w, item_his_emb)
        
        atten = tf.matmul(interest_emb, tf.reshape(item_eb, [-1, self.dim, 1]))
        atten = tf.nn.softmax(tf.pow(tf.reshape(atten, [-1, self.num_interest]), 1))
        
        readout = tf.gather(tf.reshape(interest_emb, [-1, self.dim]), tf.argmax(atten, axis=1, output_type=tf.int32) + tf.range(tf.shape(item_his_emb)[0]) * self.num_interest)
        return readout
        
class CapsuleNetwork(tf.keras.layers.Layer):
    def __init__(self, dim, seq_len, bilinear_type=1, initializer=None, num_interest=4, hard_readout=True, relu_layer=False):
        super(CapsuleNetwork, self).__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.bilinear_type = bilinear_type
        self.initializer = initializer
        self.num_interest = num_interest
        self.hard_readout = hard_readout
        self.relu_layer = relu_layer
        self.stop_grad = False

    def call(self, item_his_emb, item_eb, mask):
        with tf.compat.v1.variable_scope('bilinear', initializer=self.initializer):
            if self.bilinear_type == 0:
                item_emb_hat = tf.keras.layers.Dense(self.dim, kernel_initializer=self.initializer)(item_his_emb)
                item_emb_hat = tf.tile(item_emb_hat, [1, 1, self.num_interest])
            elif self.bilinear_type == 1:
                item_emb_hat = tf.keras.layers.Dense(self.dim * self.num_interest, kernel_initializer=self.initializer)(item_his_emb)
            else:
                w = tf.compat.v1.get_variable(
                    'weights', shape=[1, self.seq_len, self.num_interest * self.dim, self.dim],
                    initializer=self.initializer)
                # [N, T, 1, C]
                u = tf.expand_dims(item_his_emb, axis=2)
                # [N, T, num_caps * dim_caps]
                item_emb_hat = tf.reduce_sum(w[:, :self.seq_len, :, :] * u, axis=3)

        item_emb_hat = tf.reshape(item_emb_hat, [-1, self.seq_len, self.num_interest, self.dim])
        item_emb_hat = tf.transpose(item_emb_hat, [0, 2, 1, 3])
        item_emb_hat = tf.reshape(item_emb_hat, [-1, self.num_interest, self.seq_len, self.dim])

        if self.stop_grad:
            item_emb_hat_iter = tf.stop_gradient(item_emb_hat, name='item_emb_hat_iter')
        else:
            item_emb_hat_iter = item_emb_hat

        if self.bilinear_type > 0:
            # capsule_weight = tf.stop_gradient(tf.zeros([get_shape(item_his_emb)[0], self.num_interest, self.seq_len]))
            capsule_weight = tf.zeros([get_shape(item_his_emb)[0], self.num_interest, self.seq_len])
        else:
            # capsule_weight = tf.stop_gradient(tf.truncated_normal([get_shape(item_his_emb)[0], self.num_interest, self.seq_len], stddev=1.0))
            capsule_weight = tf.truncated_normal([get_shape(item_his_emb)[0], self.num_interest, self.seq_len], stddev=1.0)

        for i in range(3):
            atten_mask = tf.tile(tf.expand_dims(mask, axis=1), [1, self.num_interest, 1])
            paddings = tf.zeros_like(atten_mask, dtype=tf.float32)

            capsule_softmax_weight = tf.nn.softmax(capsule_weight, axis=1)
            capsule_softmax_weight = tf.where(tf.math.equal(atten_mask, 0), paddings, capsule_softmax_weight)
            # capsule_softmax_weight = tf.where(atten_mask==0, paddings, capsule_softmax_weight)
            capsule_softmax_weight = tf.expand_dims(capsule_softmax_weight, 2)

            if i < 2:
                interest_capsule = tf.matmul(capsule_softmax_weight, item_emb_hat_iter)
                cap_norm = tf.reduce_sum(tf.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / tf.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

                delta_weight = tf.matmul(item_emb_hat_iter, tf.transpose(interest_capsule, [0, 1, 3, 2]))
                delta_weight = tf.reshape(delta_weight, [-1, self.num_interest, self.seq_len])
                capsule_weight = capsule_weight + delta_weight
            else:
                interest_capsule = tf.matmul(capsule_softmax_weight, item_emb_hat)
                cap_norm = tf.reduce_sum(tf.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / tf.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

        interest_capsule = tf.reshape(interest_capsule, [-1, self.num_interest, self.dim])

        if self.relu_layer:
            interest_capsule = tf.keras.layers.Dense(self.dim, activation=tf.nn.relu, name='proj')(interest_capsule)

        atten = tf.matmul(interest_capsule, tf.reshape(item_eb, [-1, self.dim, 1]))
        atten = tf.nn.softmax(tf.pow(tf.reshape(atten, [-1, self.num_interest]), 1))

        if self.hard_readout:
            readout = tf.gather(tf.reshape(interest_capsule, [-1, self.dim]), tf.argmax(atten, axis=1, output_type=tf.int32) + tf.range(tf.shape(item_his_emb)[0]) * self.num_interest)
        else:
            readout = tf.matmul(tf.reshape(atten, [get_shape(item_his_emb)[0], 1, self.num_interest]), interest_capsule)
            readout = tf.reshape(readout, [get_shape(item_his_emb)[0], self.dim])

        return interest_capsule, readout