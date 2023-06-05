# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import numpy as np
import random
from collections import defaultdict

from recommenders.models.deeprec.io.iterator import BaseIterator
from recommenders.models.deeprec.deeprec_utils import load_dict, gen_fatigue_features
from recommenders.utils.constants import SEED


__all__ = ["SequentialIterator"]


class SequentialIterator(BaseIterator):
    def __init__(self, hparams, graph, col_spliter="\t"):
        """Initialize an iterator. Create necessary placeholders for the model.

        Args:
            hparams (object): Global hyper-parameters. Some key settings such as #_feature and #_field are there.
            graph (object): The running graph. All created placeholder will be added to this graph.
            col_spliter (str): Column splitter in one line.
        """
        # tf.compat.v1.set_random_seed(SEED)
        # np.random.seed(SEED)
        self.hparams = hparams
        self.our_model = False
        if self.hparams.model_type in ['model']:
            self.our_model = True
        self.col_spliter = col_spliter
        user_vocab, item_vocab, cate_vocab = (
            hparams.user_vocab,
            hparams.item_vocab,
            hparams.cate_vocab,
        )
        self.userdict, self.itemdict, self.catedict = (
            load_dict(user_vocab),
            load_dict(item_vocab),
            load_dict(cate_vocab),
        )

        self.max_seq_length = hparams.max_seq_length
        self.batch_size = hparams.batch_size
        self.iter_data = dict()

        self.graph = graph
        with self.graph.as_default():
            self.labels = tf.compat.v1.placeholder(tf.float32, [None, 1], name="label")
            self.users = tf.compat.v1.placeholder(tf.int32, [None], name="users")
            self.items = tf.compat.v1.placeholder(tf.int32, [None], name="items")
            self.cates = tf.compat.v1.placeholder(tf.int32, [None], name="cates")
            self.item_history = tf.compat.v1.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_history"
            )
            self.item_cate_history = tf.compat.v1.placeholder(
                tf.int32, [None, self.max_seq_length], name="item_cate_history"
            )
            self.mask = tf.compat.v1.placeholder(
                tf.int32, [None, self.max_seq_length], name="mask"
            )
            self.time = tf.compat.v1.placeholder(tf.float32, [None], name="time")
            self.time_diff = tf.compat.v1.placeholder(
                tf.float32, [None, self.max_seq_length], name="time_diff"
            )
            self.time_from_first_action = tf.compat.v1.placeholder(
                tf.float32, [None, self.max_seq_length], name="time_from_first_action"
            )
            self.time_to_now = tf.compat.v1.placeholder(
                tf.float32, [None, self.max_seq_length], name="time_to_now"
            )
            self.time_to_now_sec = tf.compat.v1.placeholder(
                tf.float32, [None, self.max_seq_length], name="time_to_now_sec"
            )
            if hparams.model_type in ['model']:
                self.recent_idx = tf.compat.v1.placeholder(
                    tf.int32, [None, hparams.recent_k], name="recent_idx"
                )
                self.CL_mask = tf.compat.v1.placeholder(
                    tf.float32, [None, 1], name="CL_mask"
                )
                self.CL_fatigue_mask = tf.compat.v1.placeholder(
                    tf.float32, [None, 1], name="CL_fatigue_mask"
                )
                self.users_fatigue = tf.compat.v1.placeholder(
                    tf.int32, [None], name="users_fatigue"
                )
                self.items_fatigue = tf.compat.v1.placeholder(
                    tf.int32, [None], name="items_fatigue"
                )
                self.cates_fatigue = tf.compat.v1.placeholder(
                    tf.int32, [None], name="cates_fatigue"
                )
                self.item_fatigue_history = tf.compat.v1.placeholder(
                    tf.int32, [None, self.max_seq_length], name="item_fatigue_history"
                )
                self.item_fatigue_cate_history = tf.compat.v1.placeholder(
                    tf.int32, [None, self.max_seq_length], name="item_fatigue_history"
                )
                self.recent_fatigue_idx = tf.compat.v1.placeholder(
                    tf.int32, [None, hparams.recent_k], name="recent_fatigue_idx"
                )
                self.fatigue_mask = tf.compat.v1.placeholder(
                    tf.int32, [None, self.max_seq_length], name="fatigue_mask"
                )
                self.fatigue_time_from_first_action = tf.compat.v1.placeholder(
                    tf.float32, [None, self.max_seq_length], name="fatigue_time_from_first_action"
                )
                self.fatigue_time_to_now = tf.compat.v1.placeholder(
                    tf.float32, [None, self.max_seq_length], name="fatigue_time_to_now"
                )
            if self.hparams.model_type =='dfn':
                self.fatigue_features = tf.compat.v1.placeholder(
                    tf.float32, [None, 6], name="fatigue_features"
                )

    def parse_file(self, input_file):
        """Parse the file to A list ready to be used for downstream tasks.

        Args:
            input_file: One of train, valid or test file which has never been parsed.

        Returns:
            list: A list with parsing result.
        """
        with open(input_file, "r") as f:
            lines = f.readlines()
        res = []
        for line in lines:
            if not line:
                continue
            res.append(self.parser_one_line(line))
        return res

    def parser_one_line(self, line):
        """Parse one string line into feature values.

        Args:
            line (str): a string indicating one instance.
                This string contains tab-separated values including:
                label, user_hash, item_hash, item_cate, operation_time, item_history_sequence,
                item_cate_history_sequence, and time_history_sequence.

        Returns:
            list: Parsed results including `label`, `user_id`, `item_id`, `item_cate`, `item_history_sequence`, `cate_history_sequence`,
            `current_time`, `time_diff`, `time_from_first_action`, `time_to_now`.

        """
        words = line.strip().split(self.col_spliter)
        label = int(words[0])
        user_id = self.userdict[words[1]] if words[1] in self.userdict else 0
        item_id = self.itemdict[words[2]] if words[2] in self.itemdict else 0
        item_cate = self.catedict[words[3]] if words[3] in self.catedict else 0
        current_time = float(words[4])

        item_history_sequence = []
        cate_history_sequence = []
        time_history_sequence = []

        item_history_words = words[5].strip().split(",")
        for item in item_history_words:
            item_history_sequence.append(
                self.itemdict[item] if item in self.itemdict else 0
            )

        cate_history_words = words[6].strip().split(",")
        for cate in cate_history_words:
            cate_history_sequence.append(
                self.catedict[cate] if cate in self.catedict else 0
            )

        time_history_words = words[7].strip().split(",")
        time_history_sequence = [float(i) for i in time_history_words]

        time_range = 3600 * 24

        time_diff = []
        for i in range(len(time_history_sequence) - 1):
            diff = (
                time_history_sequence[i + 1] - time_history_sequence[i]
            ) / time_range
            diff = max(diff, 0.5)
            time_diff.append(diff)
        last_diff = (current_time - time_history_sequence[-1]) / time_range
        last_diff = max(last_diff, 0.5)
        time_diff.append(last_diff)
        time_diff = np.log(time_diff)

        time_from_first_action = []
        first_time = time_history_sequence[0]
        time_from_first_action = [
            (t - first_time) / time_range for t in time_history_sequence[1:]
        ]
        time_from_first_action = [max(t, 0.5) for t in time_from_first_action]
        last_diff = (current_time - first_time) / time_range
        last_diff = max(last_diff, 0.5)
        time_from_first_action.append(last_diff)
        time_from_first_action = np.log(time_from_first_action)

        time_to_now = []
        time_to_now = [(current_time - t) / time_range for t in time_history_sequence]
        time_to_now = [max(t, 0.5) for t in time_to_now]
        time_to_now = np.log(time_to_now)
        
        time_to_now_sec = []
        time_to_now_sec = [current_time - t for t in time_history_sequence]

        return (
            label,
            user_id,
            item_id,
            item_cate,
            item_history_sequence,
            cate_history_sequence,
            current_time,
            time_diff,
            time_from_first_action,
            time_to_now,
            time_to_now_sec
        )

        
    def load_data_from_file(self, infile, batch_num_ngs=0, min_seq_length=1):
        """Read and parse data from a file.

        Args:
            infile (str): Text input file. Each line in this file is an instance.
            batch_num_ngs (int): The number of negative sampling here in batch.
                0 represents that there is no need to do negative sampling here.
            min_seq_length (int): The minimum number of a sequence length.
                Sequences with length lower than min_seq_length will be ignored.

        Yields:
            object: An iterator that yields parsed results, in the format of graph `feed_dict`.
        """
        self.infile = infile
        batch_size = self.batch_size
        if 'train' not in infile:
            batch_size *= (1+self.hparams.train_num_ngs)
        label_list = []
        user_list = []
        item_list = []
        item_cate_list = []
        item_history_batch = []
        item_cate_history_batch = []
        time_list = []
        time_diff_list = []
        time_from_first_action_list = []
        time_to_now_list = []
        time_to_now_sec_list = []

        cnt = 0

        if infile not in self.iter_data:
            lines = self.parse_file(infile)
            self.iter_data[infile] = lines
        else:
            lines = self.iter_data[infile]
        
        self.is_train = False
        if 'train' in infile:
            self.is_train = True

        if batch_num_ngs > 0:
            random.shuffle(lines)

        for line in lines:
            if not line:
                continue

            (
                label,
                user_id,
                item_id,
                item_cate,
                item_history_sequence,
                item_cate_history_sequence,
                current_time,
                time_diff,
                time_from_first_action,
                time_to_now,
                time_to_now_sec
            ) = line
            if len(item_history_sequence) < min_seq_length:
                continue

            label_list.append(label)
            user_list.append(user_id)
            item_list.append(item_id)
            item_cate_list.append(item_cate)
            item_history_batch.append(item_history_sequence)
            item_cate_history_batch.append(item_cate_history_sequence)
            time_list.append(current_time)
            time_diff_list.append(time_diff)
            time_from_first_action_list.append(time_from_first_action)
            time_to_now_list.append(time_to_now)
            time_to_now_sec_list.append(time_to_now_sec)

            cnt += 1
            if cnt == batch_size:
                res = self._convert_data(
                    label_list,
                    user_list,
                    item_list,
                    item_cate_list,
                    item_history_batch,
                    item_cate_history_batch,
                    time_list,
                    time_diff_list,
                    time_from_first_action_list,
                    time_to_now_list,
                    time_to_now_sec_list,
                    batch_num_ngs
                )
                batch_input = self.gen_feed_dict(res)
                yield batch_input if batch_input else None
                label_list = []
                user_list = []
                item_list = []
                item_cate_list = []
                item_history_batch = []
                item_cate_history_batch = []
                time_list = []
                time_diff_list = []
                time_from_first_action_list = []
                time_to_now_list = []
                time_to_now_sec_list = []
                cnt = 0
        if cnt > 0:
            res = self._convert_data(
                label_list,
                user_list,
                item_list,
                item_cate_list,
                item_history_batch,
                item_cate_history_batch,
                time_list,
                time_diff_list,
                time_from_first_action_list,
                time_to_now_list,
                time_to_now_sec_list,
                batch_num_ngs
            )
            batch_input = self.gen_feed_dict(res)
            yield batch_input if batch_input else None

    def _convert_data(
        self,
        label_list,
        user_list,
        item_list,
        item_cate_list,
        item_history_batch,
        item_cate_history_batch,
        time_list,
        time_diff_list,
        time_from_first_action_list,
        time_to_now_list,
        time_to_now_sec_list,
        batch_num_ngs
    ):
        """Convert data into numpy arrays that are good for further model operation.

        Args:
            label_list (list): A list of ground-truth labels.
            user_list (list): A list of user indexes.
            item_list (list): A list of item indexes.
            item_cate_list (list): A list of category indexes.
            item_history_batch (list): A list of item history indexes.
            item_cate_history_batch (list): A list of category history indexes.
            time_list (list): A list of current timestamp.
            time_diff_list (list): A list of timestamp between each sequential operations.
            time_from_first_action_list (list): A list of timestamp from the first operation.
            time_to_now_list (list): A list of timestamp to the current time.
            batch_num_ngs (int): The number of negative sampling while training in mini-batch.

        Returns:
            dict: A dictionary, containing multiple numpy arrays that are convenient for further operation.
        """
        if batch_num_ngs:
            instance_cnt = len(label_list)
            if instance_cnt < 5:
                return

            label_list_all = []
            item_list_all = []
            item_cate_list_all = []
            user_list_all = np.asarray(
                [[user] * (batch_num_ngs + 1) for user in user_list], dtype=np.int32
            ).flatten()
            time_list_all = np.asarray(
                [[t] * (batch_num_ngs + 1) for t in time_list], dtype=np.float32
            ).flatten()
            if self.hparams.model_type =='dfn':
                fatigue_features = []

            history_lengths = [len(item_history_batch[i]) for i in range(instance_cnt)]
            max_seq_length_batch = self.max_seq_length
            item_history_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)
            ).astype("int32")
            item_cate_history_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)
            ).astype("int32")
            time_diff_batch = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)
            ).astype("float32")
            time_from_first_action_batch = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)
            ).astype("float32")
            time_to_now_batch = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)
            ).astype("float32")
            time_to_now_sec_batch = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length_batch)
            ).astype("float32")
            mask = np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length_batch)
            ).astype("float32")
            if self.our_model:
                recent_idx = -np.ones(
                    (instance_cnt * (1 + batch_num_ngs), self.hparams.recent_k)
                ).astype("int32")
                CL_mask = np.ones(
                    (instance_cnt * (1 + batch_num_ngs), 1)
                ).astype("float32")
                users_fatigue_all = np.repeat(np.array(user_list, dtype=np.int32), batch_num_ngs)
                items_fatigue_all = np.repeat(np.array(item_list, dtype=np.int32), batch_num_ngs)
                cates_fatigue_all = np.repeat(np.array(item_cate_list, dtype=np.int32), batch_num_ngs)
                item_fatigue_history_batch_all = np.zeros(
                    (instance_cnt * batch_num_ngs, max_seq_length_batch)
                ).astype("int32")
                item_fatigue_cate_history_batch_all = np.zeros(
                    (instance_cnt * batch_num_ngs, max_seq_length_batch)
                ).astype("int32")

            for i in range(instance_cnt):
                this_length = min(history_lengths[i], max_seq_length_batch)
                for index in range(batch_num_ngs + 1):
                    item_history_batch_all[
                        i * (batch_num_ngs + 1) + index, :this_length
                    ] = np.asarray(item_history_batch[i][-this_length:], dtype=np.int32)
                    item_cate_history_batch_all[
                        i * (batch_num_ngs + 1) + index, :this_length
                    ] = np.asarray(
                        item_cate_history_batch[i][-this_length:], dtype=np.int32
                    )
                    mask[i * (batch_num_ngs + 1) + index, :this_length] = 1.0
                    time_diff_batch[
                        i * (batch_num_ngs + 1) + index, :this_length
                    ] = np.asarray(time_diff_list[i][-this_length:], dtype=np.float32)
                    time_from_first_action_batch[
                        i * (batch_num_ngs + 1) + index, :this_length
                    ] = np.asarray(
                        time_from_first_action_list[i][-this_length:], dtype=np.float32
                    )
                    time_to_now_batch[
                        i * (batch_num_ngs + 1) + index, :this_length
                    ] = np.asarray(time_to_now_list[i][-this_length:], dtype=np.float32)
                    time_to_now_sec_batch[
                        i * (batch_num_ngs + 1) + index, :this_length
                    ] = np.asarray(time_to_now_sec_list[i][-this_length:], dtype=np.float32)
                    if self.our_model:
                        tmp_idx = np.arange(max(0, this_length-self.hparams.recent_k), this_length)
                        recent_idx[i * (batch_num_ngs + 1) + index, :len(tmp_idx)] = tmp_idx
                        
                        first_idx = max(0, this_length-self.hparams.recent_k)
                        target_cate, target_item = item_cate_list[i], item_list[i]
                        item_fatigue_seq, cate_fatigue_seq = np.array(item_history_batch[i][-this_length:]), np.array(item_cate_history_batch[i][-this_length:])
                        recent_seq = item_fatigue_seq[first_idx:]
                        if (this_length >= self.hparams.CL_thr) and ((recent_seq==target_item).mean() <= 0.5):
                            replace_idx = np.where(recent_seq!=target_item)[0]
                            replace_idx = np.random.choice(replace_idx, np.random.choice(np.arange(max(1, (recent_seq==target_item).sum()), len(replace_idx)+1)), replace=False)
                            item_fatigue_seq[first_idx+replace_idx], cate_fatigue_seq[first_idx+replace_idx] = target_item, target_cate
                        else:
                            CL_mask[i * (batch_num_ngs + 1) + index, 0] = 0
                        
                        if self.is_train and (index < batch_num_ngs):
                            item_fatigue_history_batch_all[i * batch_num_ngs + index, :this_length] = item_fatigue_seq
                            item_fatigue_cate_history_batch_all[i * batch_num_ngs + index, :this_length] = cate_fatigue_seq

            for i in range(instance_cnt):
                this_length = min(history_lengths[i], max_seq_length_batch)
                positive_item = item_list[i]
                label_list_all.append(1)
                item_list_all.append(positive_item)
                item_cate_list_all.append(item_cate_list[i])
                if self.hparams.model_type =='dfn':
                    fatigue_features.append(gen_fatigue_features(np.array(item_cate_history_batch[i][-this_length:]), item_cate_list[i], np.array(time_to_now_sec_list[i][-this_length:])))
                count = 0
                while batch_num_ngs:
                    random_value = random.randint(0, instance_cnt - 1)
                    negative_item = item_list[random_value]
                    if negative_item == positive_item:
                        continue
                    label_list_all.append(0)
                    item_list_all.append(negative_item)
                    item_cate_list_all.append(item_cate_list[random_value])
                    if self.hparams.model_type =='dfn':
                        fatigue_features.append(gen_fatigue_features(np.array(item_cate_history_batch[i][-this_length:]), item_cate_list[random_value], np.array(time_to_now_sec_list[i][-this_length:])))
                    count += 1
                    if count == batch_num_ngs:
                        break

            res = {}
            res["labels"] = np.asarray(label_list_all, dtype=np.float32).reshape(-1, 1)
            res["users"] = user_list_all
            res["items"] = np.asarray(item_list_all, dtype=np.int32)
            res["cates"] = np.asarray(item_cate_list_all, dtype=np.int32)
            res["item_history"] = item_history_batch_all
            res["item_cate_history"] = item_cate_history_batch_all
            res["mask"] = mask
            res["time"] = time_list_all
            res["time_diff"] = time_diff_batch
            res["time_from_first_action"] = time_from_first_action_batch
            res["time_to_now"] = time_to_now_batch
            res["time_to_now_sec"] = time_to_now_sec_batch
            if self.our_model:
                res['recent_idx'] = recent_idx
                res['CL_mask'] = CL_mask
                res['users_fatigue'] = users_fatigue_all
                res['items_fatigue'] = items_fatigue_all
                res['cates_fatigue'] = cates_fatigue_all
                res['item_fatigue_history'] = item_fatigue_history_batch_all
                res['item_fatigue_cate_history'] = item_fatigue_cate_history_batch_all
            if self.hparams.model_type =='dfn':
                res['fatigue_features'] = np.asarray(fatigue_features, dtype=np.float32)
            return res

        else:
            instance_cnt = len(label_list)
            neg_idx = np.array(label_list)==0
            negative_count = np.sum(neg_idx)
            history_lengths = [len(item_history_batch[i]) for i in range(instance_cnt)]
            max_seq_length_batch = self.max_seq_length
            item_history_batch_all = np.zeros(
                (instance_cnt, max_seq_length_batch)
            ).astype("int32")
            item_cate_history_batch_all = np.zeros(
                (instance_cnt, max_seq_length_batch)
            ).astype("int32")
            time_diff_batch = np.zeros((instance_cnt, max_seq_length_batch)).astype(
                "float32"
            )
            time_from_first_action_batch = np.zeros(
                (instance_cnt, max_seq_length_batch)
            ).astype("float32")
            time_to_now_batch = np.zeros((instance_cnt, max_seq_length_batch)).astype(
                "float32"
            )
            time_to_now_sec_batch = np.zeros((instance_cnt, max_seq_length_batch)).astype(
                "float32"
            )
            mask = np.zeros((instance_cnt, max_seq_length_batch)).astype("float32")
            if self.hparams.model_type =='dfn':
                fatigue_features = []
            if self.our_model:
                recent_idx = -np.ones(
                    (instance_cnt, self.hparams.recent_k)
                ).astype("int32")
                CL_mask = np.ones(
                    (instance_cnt, 1)
                ).astype("float32")
                users_fatigue_all = np.array(user_list, dtype=np.int32)[neg_idx]
                items_fatigue_all = np.array(item_list, dtype=np.int32)[neg_idx]
                cates_fatigue_all = np.array(item_cate_list, dtype=np.int32)[neg_idx]
                item_fatigue_history_batch_all = np.zeros(
                    (negative_count, max_seq_length_batch)
                ).astype("int32")
                item_fatigue_cate_history_batch_all = np.zeros(
                    (negative_count, max_seq_length_batch)
                ).astype("int32")

            neg_cnt = 0
            for i in range(instance_cnt):
                this_length = min(history_lengths[i], max_seq_length_batch)
                item_history_batch_all[i, :this_length] = item_history_batch[i][
                    -this_length:
                ]
                item_cate_history_batch_all[i, :this_length] = item_cate_history_batch[
                    i
                ][-this_length:]
                mask[i, :this_length] = 1.0
                time_diff_batch[i, :this_length] = time_diff_list[i][-this_length:]
                time_from_first_action_batch[
                    i, :this_length
                ] = time_from_first_action_list[i][-this_length:]
                time_to_now_batch[i, :this_length] = time_to_now_list[i][-this_length:]
                time_to_now_sec_batch[i, :this_length] = time_to_now_sec_list[i][-this_length:]
                if self.our_model:
                    tmp_idx = np.arange(max(0, this_length-self.hparams.recent_k), this_length)
                    recent_idx[i, :len(tmp_idx)] = tmp_idx
                    
                    first_idx = max(0, this_length-self.hparams.recent_k)
                    if label_list[i] > 0:
                        target_cate, target_item = item_cate_list[i], item_list[i]
                    item_fatigue_seq, cate_fatigue_seq = np.array(item_history_batch[i][-this_length:]), np.array(item_cate_history_batch[i][-this_length:])
                    recent_seq = item_fatigue_seq[first_idx:]
                    if (this_length >= self.hparams.CL_thr) and ((recent_seq==target_item).mean() <= 0.5):
                        replace_idx = np.where(recent_seq!=target_item)[0]
                        replace_idx = np.random.choice(replace_idx, np.random.choice(np.arange(max(1, (recent_seq==target_item).sum()), len(replace_idx)+1)), replace=False)
                        item_fatigue_seq[first_idx+replace_idx], cate_fatigue_seq[first_idx+replace_idx] = target_item, target_cate
                    else:
                        CL_mask[i, 0] = 0
                        
                    if self.is_train and (label_list[i]==0):
                        item_fatigue_history_batch_all[neg_cnt, :this_length] = item_fatigue_seq
                        item_fatigue_cate_history_batch_all[neg_cnt, :this_length] = cate_fatigue_seq
                        neg_cnt += 1
                if self.hparams.model_type =='dfn':
                    fatigue_features.append(gen_fatigue_features(np.array(item_cate_history_batch[i][-this_length:]), item_cate_list[i], np.array(time_to_now_sec_list[i][-this_length:])))
                

            res = {}
            res["labels"] = np.asarray(label_list, dtype=np.float32).reshape(-1, 1)
            res["users"] = np.asarray(user_list, dtype=np.float32)
            res["items"] = np.asarray(item_list, dtype=np.int32)
            res["cates"] = np.asarray(item_cate_list, dtype=np.int32)
            res["item_history"] = item_history_batch_all
            res["item_cate_history"] = item_cate_history_batch_all
            res["mask"] = mask
            res["time"] = np.asarray(time_list, dtype=np.float32)
            res["time_diff"] = time_diff_batch
            res["time_from_first_action"] = time_from_first_action_batch
            res["time_to_now"] = time_to_now_batch
            res["time_to_now_sec"] = time_to_now_sec_batch
            if self.our_model:
                res['recent_idx'] = recent_idx
                res['CL_mask'] = CL_mask
                res['users_fatigue'] = users_fatigue_all
                res['items_fatigue'] = items_fatigue_all
                res['cates_fatigue'] = cates_fatigue_all
                res['item_fatigue_history'] = item_fatigue_history_batch_all
                res['item_fatigue_cate_history'] = item_fatigue_cate_history_batch_all
            if self.hparams.model_type =='dfn':
                res['fatigue_features'] = np.asarray(fatigue_features, dtype=np.float32)
            return res

    def gen_feed_dict(self, data_dict):
        """Construct a dictionary that maps graph elements to values.

        Args:
            data_dict (dict): A dictionary that maps string name to numpy arrays.

        Returns:
            dict: A dictionary that maps graph elements to numpy arrays.

        """
        if not data_dict:
            return dict()
        feed_dict = {
            self.labels: data_dict["labels"],
            self.users: data_dict["users"],
            self.items: data_dict["items"],
            self.cates: data_dict["cates"],
            self.item_history: data_dict["item_history"],
            self.item_cate_history: data_dict["item_cate_history"],
            self.mask: data_dict["mask"],
            self.time: data_dict["time"],
            self.time_diff: data_dict["time_diff"],
            self.time_from_first_action: data_dict["time_from_first_action"],
            self.time_to_now: data_dict["time_to_now"],
            self.time_to_now_sec: data_dict["time_to_now_sec"]
        }
        if self.our_model:
            feed_dict[self.recent_idx] = data_dict["recent_idx"]
            feed_dict[self.CL_mask] = data_dict["CL_mask"]
            feed_dict[self.CL_fatigue_mask] = data_dict["CL_mask"][data_dict['labels'][:, 0]==0]
            feed_dict[self.users_fatigue] = data_dict["users_fatigue"]
            feed_dict[self.items_fatigue] = data_dict["items_fatigue"]
            feed_dict[self.cates_fatigue] = data_dict["cates_fatigue"]
            feed_dict[self.item_fatigue_history] = data_dict["item_fatigue_history"]
            feed_dict[self.item_fatigue_cate_history] = data_dict["item_fatigue_cate_history"]
            feed_dict[self.recent_fatigue_idx] = data_dict["recent_idx"][data_dict['labels'][:, 0]==0]
            feed_dict[self.fatigue_mask] = data_dict["mask"][data_dict['labels'][:, 0]==0]
            feed_dict[self.fatigue_time_from_first_action] = data_dict["time_from_first_action"][data_dict['labels'][:, 0]==0]
            feed_dict[self.fatigue_time_to_now] = data_dict["time_to_now"][data_dict['labels'][:, 0]==0]
        if self.hparams.model_type =='dfn':
            feed_dict[self.fatigue_features] = data_dict["fatigue_features"]
        return feed_dict
