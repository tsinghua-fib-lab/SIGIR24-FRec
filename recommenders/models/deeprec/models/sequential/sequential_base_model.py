# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import os
import abc
import numpy as np
import tensorflow as tf
import pandas as pd

from recommenders.models.deeprec.models.base_model import BaseModel
from recommenders.models.deeprec.deeprec_utils import cal_metric, load_dict


__all__ = ["SequentialBaseModel"]


class SequentialBaseModel(BaseModel):
    """Base class for sequential models"""

    def __init__(self, hparams, iterator_creator, graph=None, seed=None):
        """Initializing the model. Create common logics which are needed by all sequential models, such as loss function,
        parameter set.

        Args:
            hparams (HParams): A `HParams` object, hold the entire set of hyperparameters.
            iterator_creator (object): An iterator to load the data.
            graph (object): An optional graph.
            seed (int): Random seed.
        """
        self.hparams = hparams
        self.need_sample = hparams.need_sample
        self.train_num_ngs = hparams.train_num_ngs
        if self.train_num_ngs is None:
            raise ValueError(
                "Please confirm the number of negative samples for each positive instance."
            )
        self.min_seq_length = (
            hparams.min_seq_length if "min_seq_length" in hparams.values() else 1
        )
        self.hidden_size = (
            hparams.hidden_size if "hidden_size" in hparams.values() else None
        )
        self.graph = tf.Graph() if not graph else graph

        with self.graph.as_default():
            self.sequence_length = tf.compat.v1.placeholder(
                tf.int32, [None], name="sequence_length"
            )
            # self.embedding_keeps = tf.compat.v1.placeholder(tf.float32, name="embedding_keeps")
            # self.embedding_keep_prob_train = None
            # self.embedding_keep_prob_test = None

        super().__init__(hparams, iterator_creator, graph=self.graph, seed=seed)

    @abc.abstractmethod
    def _build_seq_graph(self):
        """Subclass will implement this."""
        pass

    def _build_graph(self):
        """The main function to create sequential models.

        Returns:
            object: the prediction score make by the model.
        """
        hparams = self.hparams
        self.keep_prob_train = 1 - np.array(hparams.dropout)
        self.keep_prob_test = np.ones_like(hparams.dropout)

        with tf.compat.v1.variable_scope("sequential") as self.sequential_scope:
            self._build_embedding()
            self._lookup_from_embedding()
            if self.our_model:
                interest_output_rec, fatigue_logit, interest_output_fatigue, fatigue_logit_fatigue = self._build_seq_graph()
            else:
                model_output_rec = self._build_seq_graph()
            if hparams.model_type in ['sasrec']:
                logit = tf.reduce_sum(model_output_rec*self.target_item_embedding, -1, True)
            elif hparams.model_type in ['mgnm']:
                logit = tf.concat([tf.reduce_sum(user_emb*self.target_item_embedding, -1, True) for user_emb in model_output_rec], -1)
            elif self.our_model:
                interest_logit = self._fcn_net(interest_output_rec, hparams.layer_sizes, scope="interest_fcn")
                interest_logit = interest_logit - tf.nn.tanh(fatigue_logit)*self.iterator.CL_mask
            else:
                logit = self._fcn_net(model_output_rec, hparams.layer_sizes, scope="logit_fcn")
            self._add_norm()
            if self.our_model:
                return interest_logit, fatigue_logit, fatigue_logit_fatigue
            else:
                return logit

    def fit(
        self,
        train_file,
        valid_file,
        valid_num_ngs,
        eval_metric="group_auc",
    ):
        """Fit the model with `train_file`. Evaluate the model on `valid_file` per epoch to observe the training status.
        If `test_file` is not None, evaluate it too.

        Args:
            train_file (str): training data set.
            valid_file (str): validation set.
            valid_num_ngs (int): the number of negative instances with one positive instance in validation data.
            eval_metric (str): the metric that control early stopping. e.g. "auc", "group_auc", etc.

        Returns:
            object: An instance of self.
        """

        # check bad input.
        if not self.need_sample and self.train_num_ngs < 1:
            raise ValueError(
                "Please specify a positive integer of negative numbers for training without sampling needed."
            )
        if valid_num_ngs < 1:
            raise ValueError(
                "Please specify a positive integer of negative numbers for validation."
            )

        if self.need_sample and self.train_num_ngs < 1:
            self.train_num_ngs = 1

        if self.hparams.write_tfevents and self.hparams.SUMMARIES_DIR:
            if not os.path.exists(self.hparams.SUMMARIES_DIR):
                os.makedirs(self.hparams.SUMMARIES_DIR)

            self.writer = tf.compat.v1.summary.FileWriter(
                self.hparams.SUMMARIES_DIR, self.sess.graph
            )

        train_sess = self.sess
        eval_info = list()

        best_metric, self.best_epoch = 0, 0
        for epoch in range(1, self.hparams.epochs + 1):
            step = 0
            self.hparams.current_epoch = epoch
            epoch_loss = 0
            file_iterator = self.iterator.load_data_from_file(
                train_file,
                self.train_num_ngs,
                self.min_seq_length,
            )

            for batch_data_input in file_iterator:
                if batch_data_input:
                    step_result = self.train(train_sess, batch_data_input)
                    if self.our_model:
                        (_, _, step_loss, step_data_loss, summary, fatigue_loss) = step_result
                    else:
                        (_, _, step_loss, step_data_loss, summary) = step_result
                    if self.hparams.write_tfevents and self.hparams.SUMMARIES_DIR:
                        self.writer.add_summary(summary, step)
                    epoch_loss += step_loss
                    step += 1
                    if step % self.hparams.show_step == 0:
                        if self.our_model:
                            print(
                                "step {0:d} , total_loss: {1:.4f}, data_loss: {2:.4f}, CL loss: {3:.4f}".format(
                                    step, step_loss, step_data_loss, fatigue_loss
                                )
                            )
                        else:
                            print(
                                "step {0:d} , total_loss: {1:.4f}, data_loss: {2:.4f}".format(
                                    step, step_loss, step_data_loss
                                )
                            )

            valid_res = self.run_eval(valid_file, valid_num_ngs)
            print(
                "eval valid at epoch {0}: {1}".format(
                    epoch,
                    ",".join(
                        [
                            "" + str(key) + ":" + str(value)
                            for key, value in valid_res.items()
                        ]
                    ),
                )
            )
            eval_info.append((epoch, valid_res))

            progress = False
            early_stop = self.hparams.EARLY_STOP
            if valid_res[eval_metric] > best_metric:
                best_metric = valid_res[eval_metric]
                self.best_epoch = epoch
                progress = True
            else:
                if early_stop > 0 and epoch - self.best_epoch >= early_stop:
                    print("early stop at epoch {0}!".format(epoch))
                    break

            if self.hparams.save_model and self.hparams.MODEL_DIR:
                if not os.path.exists(self.hparams.MODEL_DIR):
                    os.makedirs(self.hparams.MODEL_DIR)
                if epoch % self.hparams.save_epoch == 0:
                    checkpoint_path = self.saver.save(
                        sess=train_sess,
                        save_path=os.path.join(self.hparams.MODEL_DIR, "epoch_" + str(epoch)),
                    )
                if progress:
                    checkpoint_path = self.saver.save(  # noqa: F841
                        sess=train_sess,
                        save_path=os.path.join(self.hparams.MODEL_DIR, "best_model"),
                    )

        if self.hparams.write_tfevents:
            self.writer.close()

        print(eval_info)
        print("best epoch: {0}".format(self.best_epoch))
        return self

    def run_eval(self, filename, num_ngs):
        """Evaluate the given file and returns some evaluation metrics.

        Args:
            filename (str): A file name that will be evaluated.
            num_ngs (int): The number of negative sampling for a positive instance.

        Returns:
            dict: A dictionary that contains evaluation metrics.
        """

        load_sess = self.sess
        users = []
        preds = []
        labels = []
        group_preds = []
        group_labels = []
        group = num_ngs + 1

        for batch_data_input in self.iterator.load_data_from_file(
            filename, min_seq_length=self.min_seq_length, batch_num_ngs=0
        ):
            if batch_data_input:
                step_users, step_pred, step_labels = self.eval(load_sess, batch_data_input)
                users.extend(np.reshape(step_users, -1))
                preds.extend(np.reshape(step_pred, -1))
                labels.extend(np.reshape(step_labels, -1))
                group_preds.extend(np.reshape(step_pred, (-1, group)))
                group_labels.extend(np.reshape(step_labels, (-1, group)))

        res = cal_metric(users, labels, preds, self.hparams.metrics)
        res_pairwise = cal_metric(users,
            group_labels, group_preds, self.hparams.pairwise_metrics
        )
        res.update(res_pairwise)
        return res
    
    def predict(self, infile_name, outfile_name):
        load_sess = self.sess
        to_writes = []
        for batch_data_input in self.iterator.load_data_from_file(
            infile_name, batch_num_ngs=0
        ):
            if batch_data_input:
                step_pred = self.infer(load_sess, batch_data_input)
                step_pred = np.reshape(step_pred, -1)
                time_to_now, cates, item_cate_history = batch_data_input[self.iterator.time_to_now_sec], batch_data_input[self.iterator.cates], batch_data_input[self.iterator.item_cate_history]
                recent_mask = (time_to_now>0)&(time_to_now<=3600*3)
                recent_len = np.sum(recent_mask, -1)
                recent_target_len = np.sum(recent_mask*(item_cate_history==cates.reshape([-1, 1])), -1)
                to_write = pd.DataFrame.from_dict({'#recent items':recent_len, '#target category':recent_target_len, 'prediction': step_pred})
                to_writes.append(to_write)
        pd.concat(to_writes, ignore_index=True).to_csv(outfile_name, index=False, sep='\t')
        return self

    def _build_embedding(self):
        """The field embedding layer. Initialization of embedding variables."""
        hparams = self.hparams
        self.user_vocab_length = len(load_dict(hparams.user_vocab))
        self.item_vocab_length = len(load_dict(hparams.item_vocab))
        self.cate_vocab_length = len(load_dict(hparams.cate_vocab))
        self.user_embedding_dim = hparams.user_embedding_dim
        self.item_embedding_dim = hparams.item_embedding_dim
        self.cate_embedding_dim = hparams.cate_embedding_dim

        with tf.compat.v1.variable_scope("embedding", initializer=self.initializer):
            self.user_lookup = tf.compat.v1.get_variable(
                name="user_embedding",
                shape=[self.user_vocab_length, self.user_embedding_dim],
                dtype=tf.float32,
            )
            self.item_lookup = tf.compat.v1.get_variable(
                name="item_embedding",
                shape=[self.item_vocab_length, self.item_embedding_dim],
                dtype=tf.float32,
            )
            self.cate_lookup = tf.compat.v1.get_variable(
                name="cate_embedding",
                shape=[self.cate_vocab_length, self.cate_embedding_dim],
                dtype=tf.float32,
            )
            if self.hparams.model_type in ['sasrec']:
                self.pos_lookup = tf.compat.v1.get_variable(
                    name="pos_embedding",
                    shape=[self.hparams.max_seq_length, self.item_embedding_dim+self.cate_embedding_dim],
                    dtype=tf.float32,
                )
            if self.hparams.model_type in ['clsr']:
                self.user_long_lookup = tf.compat.v1.get_variable(
                    name="user_long_embedding",
                    shape=[self.user_vocab_length, self.user_embedding_dim],
                    dtype=tf.float32,
                )
                self.user_short_lookup = tf.compat.v1.get_variable(
                    name="user_short_embedding",
                    shape=[self.user_vocab_length, self.user_embedding_dim],
                    dtype=tf.float32,
                )

    def _lookup_from_embedding(self):
        """Lookup from embedding variables. A dropout layer follows lookup operations."""
        self.user_embedding = tf.nn.embedding_lookup(
            params=self.user_lookup, ids=self.iterator.users
        )
        tf.compat.v1.summary.histogram("user_embedding_output", self.user_embedding)

        self.item_embedding = tf.compat.v1.nn.embedding_lookup(
            params=self.item_lookup, ids=self.iterator.items
        )
        self.item_history_embedding = tf.compat.v1.nn.embedding_lookup(
            params=self.item_lookup, ids=self.iterator.item_history
        )
        tf.compat.v1.summary.histogram(
            "item_history_embedding_output", self.item_history_embedding
        )

        self.cate_embedding = tf.compat.v1.nn.embedding_lookup(
            params=self.cate_lookup, ids=self.iterator.cates
        )
        self.cate_history_embedding = tf.compat.v1.nn.embedding_lookup(
            params=self.cate_lookup, ids=self.iterator.item_cate_history
        )
        tf.compat.v1.summary.histogram(
            "cate_history_embedding_output", self.cate_history_embedding
        )
        
        if self.our_model:
            self.users_fatigue_embedding = tf.nn.embedding_lookup(
                params=self.user_lookup, ids=self.iterator.users_fatigue
            )
            self.items_fatigue_embedding = tf.nn.embedding_lookup(
                params=self.item_lookup, ids=self.iterator.items_fatigue
            )
            self.item_fatigue_history_embedding = tf.compat.v1.nn.embedding_lookup(
                params=self.item_lookup, ids=self.iterator.item_fatigue_history
            )
            self.cate_fatigue_embedding = tf.compat.v1.nn.embedding_lookup(
                params=self.cate_lookup, ids=self.iterator.cates_fatigue
            )
            self.cate_fatigue_history_embedding = tf.compat.v1.nn.embedding_lookup(
                params=self.cate_lookup, ids=self.iterator.item_fatigue_cate_history
            )

        involved_items = tf.concat(
            [
                tf.reshape(self.iterator.item_history, [-1]),
                tf.reshape(self.iterator.items, [-1]),
            ],
            -1,
        )
        if self.our_model and (self.hparams.alpha>0):
            involved_items = tf.concat([involved_items, \
                                        tf.reshape(self.iterator.items_fatigue, [-1]), \
                                        tf.reshape(self.iterator.item_fatigue_history, [-1])], -1)
        self.involved_items, _ = tf.unique(involved_items)
        involved_item_embedding = tf.nn.embedding_lookup(
            params=self.item_lookup, ids=self.involved_items
        )
        self.embed_params.append(involved_item_embedding)

        involved_cates = tf.concat(
            [
                tf.reshape(self.iterator.item_cate_history, [-1]),
                tf.reshape(self.iterator.cates, [-1]),
            ],
            -1,
        )
        if self.our_model and (self.hparams.alpha>0):
            involved_cates = tf.concat([involved_cates, \
                                        tf.reshape(self.iterator.cates_fatigue, [-1]), \
                                        tf.reshape(self.iterator.item_fatigue_cate_history, [-1])], -1)
        self.involved_cates, _ = tf.unique(involved_cates)
        involved_cate_embedding = tf.nn.embedding_lookup(
            params=self.cate_lookup, ids=self.involved_cates
        )
        self.embed_params.append(involved_cate_embedding)

        self.target_item_embedding = tf.concat(
            [self.item_embedding, self.cate_embedding], -1
        )
        tf.compat.v1.summary.histogram(
            "target_item_embedding_output", self.target_item_embedding
        )
        if self.our_model:
            self.target_item_fatigue_embedding = tf.concat(
                [self.items_fatigue_embedding, self.cate_fatigue_embedding], -1
            )

    def _add_norm(self):
        """Regularization for embedding variables and other variables."""
        all_variables, embed_variables = (
            tf.compat.v1.trainable_variables(),
            tf.compat.v1.trainable_variables(
                self.sequential_scope._name + "/embedding"
            ),
        )
        layer_params = list(set(all_variables) - set(embed_variables))
        layer_params = [a for a in layer_params if "_no_reg" not in a.name]
        self.layer_params.extend(layer_params)
