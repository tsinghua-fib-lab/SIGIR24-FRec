# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import sys
sys.path.append('/home/linian/.local/lib/python3.7/site-packages/recommenders/models/deeprec/models/sequential/')
import tensorflow as tf
from sli_rec import (
    SLI_RECModel,
)
from tensorflow.compat.v1.nn.rnn_cell import GRUCell
from recommenders.models.deeprec.models.sequential.rnn_cell_implement import (
    VecAttGRUCell,
)
# from tensorflow.compat.v1.nn import dynamic_rnn
from rnn_dien import dynamic_rnn

__all__ = ["DFNModel"]


class DFNModel(SLI_RECModel):

    def _build_seq_graph(self):
        """The main function to create din model.
        
        Returns:
            obj:the output of din section.
        """
        with tf.name_scope('din'):
            hist_input = tf.concat(
                [self.item_history_embedding, self.cate_history_embedding], 2
            )
            new_target_item_embedding = tf.concat([self.target_item_embedding, self.iterator.fatigue_features], -1)
            self.mask = self.iterator.mask
            self.real_mask = tf.cast(self.mask, tf.float32)
            self.hist_embedding_sum = tf.reduce_sum(hist_input*tf.expand_dims(self.real_mask, -1), 1)
            attention_output = self._attention_fcn(new_target_item_embedding, hist_input)
            att_fea = tf.reduce_sum(attention_output, 1)
            tf.summary.histogram('att_fea', att_fea)
        model_output = tf.concat([new_target_item_embedding, self.hist_embedding_sum, att_fea], -1)
        tf.summary.histogram("model_output", model_output)
        return model_output
