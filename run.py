import sys
import os
import logging
from tempfile import TemporaryDirectory
import numpy as np
import tensorflow.compat.v1 as tf
tf.get_logger().setLevel('ERROR') # only show error messages
from absl import app
from absl import flags
from decimal import Decimal

from setproctitle import setproctitle

from recommenders.utils.timer import Timer
from recommenders.utils.constants import SEED
from recommenders.models.deeprec.deeprec_utils import (
    prepare_hparams
)
from recommenders.datasets.amazon_reviews import download_and_extract, data_preprocessing
from recommenders.datasets.download_utils import maybe_download

from recommenders.models.deeprec.models.sequential.caser import CaserModel
from recommenders.models.deeprec.models.sequential.comi import Comi
from recommenders.models.deeprec.models.sequential.sli_rec import SLI_RECModel
from recommenders.models.deeprec.models.sequential.clsr import CLSRModel
from recommenders.models.deeprec.models.sequential.sum import SUMModel
from recommenders.models.deeprec.models.sequential.gru4rec import GRU4RecModel
from recommenders.models.deeprec.models.sequential.dien import DIENModel
from recommenders.models.deeprec.models.sequential.din import DINModel
from recommenders.models.deeprec.models.sequential.dfn import DFNModel
from recommenders.models.deeprec.models.sequential.sasrec import SASRec
from recommenders.models.deeprec.models.sequential.model import Model

from recommenders.models.deeprec.io.sequential_iterator import (
    SequentialIterator
)

print("System version: {}".format(sys.version))
print("Tensorflow version: {}".format(tf.__version__))

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'taobao', 'Dataset name.')
flags.DEFINE_integer('gpu_id', 0, 'GPU ID.')
flags.DEFINE_integer('train_num_ngs', 4, 'Number of negative instances with a positive instance for training.')
flags.DEFINE_integer('valid_num_ngs', 4, 'Number of negative instances with a positiver instance for validation.')
flags.DEFINE_integer('test_num_ngs', 9, 'Number of negative instances with a positive instance for testing.')
flags.DEFINE_string('name', '', 'Experiment name.')
flags.DEFINE_string('model', 'model', 'Model name.')
flags.DEFINE_boolean('only_test', False, 'Only test and do not train.')
flags.DEFINE_string('rnn_model', 'time4lstm', 'RNN model option, could be gru, lstm, time4lstm.')
flags.DEFINE_integer('epochs', 100, 'Number of epochs.')
flags.DEFINE_integer('early_stop', 2, 'Patience for early stop.')
flags.DEFINE_string('data_path', 'data/', 'Data file path.')
flags.DEFINE_string('save_path', './', 'Save file path.')
flags.DEFINE_float('embed_l2', 1e-6, 'L2 regulation for embeddings.')
flags.DEFINE_float('layer_l2', 1e-6, 'L2 regulation for layers.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
flags.DEFINE_string('sequential_model', 'time4lstm', 'Sequential model option, could be gru, lstm, time4lstm.')
flags.DEFINE_boolean('fatigue_weight', True, 'Whether fuse interest embeddings with fatigue weights')
flags.DEFINE_boolean('fatigue_emb', True, 'Whether add fatigue embedding')
flags.DEFINE_integer('num_cross_layers', 2, '#cross')
flags.DEFINE_integer('num_dense_layers', 0, '#dense')
flags.DEFINE_integer('recent_k', 40, 'Number of recent items.')
flags.DEFINE_integer('num_interests', 4, 'Number of interests.')
flags.DEFINE_integer('CL_thr', 1, 'CL_thr')
flags.DEFINE_integer('k_size', 5, 'k_size')
flags.DEFINE_float('alpha', 0.1, 'CL learning weight.')
flags.DEFINE_integer('slots', 4, 'Number of slots.')
flags.DEFINE_integer('num_blocks', 2, 'For SASRec')
flags.DEFINE_integer('attention_num_heads', 4, 'For SASRec')
flags.DEFINE_float('contrastive_loss_weight', 0.1, 'contrastive_loss_weight.')
flags.DEFINE_integer('contrastive_length_threshold', 10, 'contrastive_length_threshold.')
flags.DEFINE_integer('L', 3, 'Markov order.')
flags.DEFINE_integer('nv', 10, 'vertical dim.')
flags.DEFINE_integer('nh', 10, 'horizonal dim.')
flags.DEFINE_string('extractor', 'sa', 'caps or sa for comirec')


def get_model(flags_obj, exp_name=''):

    RANDOM_SEED = SEED  # Set None for non-deterministic result
    
    input_creator = SequentialIterator

    setproctitle(f'{flags_obj.dataset}_{flags_obj.model}_{exp_name}')
    model_path = os.path.join(flags_obj.save_path, f"saved_models/{flags_obj.dataset}/{flags_obj.model}/{exp_name}/")
    summary_path = os.path.join(flags_obj.save_path, f"saved_summaries/{flags_obj.dataset}/{flags_obj.model}/{exp_name}/")

    yaml_file = f'configs/{flags_obj.dataset}_{flags_obj.model}.yaml'
    
    if flags_obj.model == 'caser':
        hparams = prepare_hparams(yaml_file, 
                                embed_l2=flags_obj.embed_l2, 
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                L=flags_obj.L,
                                n_v=flags_obj.nv,
                                n_h=flags_obj.nh,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path
                    )
        model = CaserModel(hparams, input_creator, seed=None)

    if flags_obj.model == 'comi':
        hparams = prepare_hparams(yaml_file, 
                                embed_l2=flags_obj.embed_l2, 
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                slots=flags_obj.slots,
                                extractor=flags_obj.extractor,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path
                    )
        model = Comi(hparams, input_creator, seed=None)

    if flags_obj.model == 'slirec':
        hparams = prepare_hparams(yaml_file, 
                                embed_l2=flags_obj.embed_l2, 
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path
                    )
        model = SLI_RECModel(hparams, input_creator, seed=None)
        
    if flags_obj.model == 'clsr':
        hparams = prepare_hparams(yaml_file, 
                                embed_l2=flags_obj.embed_l2, 
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                contrastive_loss_weight=flags_obj.contrastive_loss_weight,
                                contrastive_length_threshold=flags_obj.contrastive_length_threshold,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path
                    )
        model = CLSRModel(hparams, input_creator, seed=None)
        
    if flags_obj.model == 'gru4rec':
        hparams = prepare_hparams(yaml_file, 
                                embed_l2=flags_obj.embed_l2, 
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path
                    )
        model = GRU4RecModel(hparams, input_creator, seed=None)
        
    if flags_obj.model == 'dien':
        hparams = prepare_hparams(yaml_file, 
                                embed_l2=flags_obj.embed_l2, 
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path
                    )
        model = DIENModel(hparams, input_creator, seed=None)
        
    if flags_obj.model == 'din':
        hparams = prepare_hparams(yaml_file, 
                                embed_l2=flags_obj.embed_l2, 
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path
                    )
        model = DINModel(hparams, input_creator, seed=None)
        
    if flags_obj.model == 'dfn':
        hparams = prepare_hparams(yaml_file, 
                                embed_l2=flags_obj.embed_l2, 
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path
                    )
        model = DFNModel(hparams, input_creator, seed=None)
        
    if flags_obj.model == 'sasrec':
        hparams = prepare_hparams(yaml_file, 
                                embed_l2=flags_obj.embed_l2, 
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                num_blocks=flags_obj.num_blocks,
                                attention_num_heads=flags_obj.attention_num_heads,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path
                    )
        model = SASRec(hparams, input_creator, seed=None)
        
    if flags_obj.model == 'sum':
        hparams = prepare_hparams(yaml_file, 
                                embed_l2=flags_obj.embed_l2, 
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                slots=flags_obj.slots,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path
                    )
        model = SUMModel(hparams, input_creator, seed=None)
        
    if flags_obj.model == 'model':
        hparams = prepare_hparams(yaml_file, 
                                embed_l2=flags_obj.embed_l2, 
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                fatigue_weight=flags_obj.fatigue_weight,
                                fatigue_emb=flags_obj.fatigue_emb,
                                num_cross_layers=flags_obj.num_cross_layers,
                                num_dense_layers=flags_obj.num_dense_layers,
                                recent_k=flags_obj.recent_k,
                                num_interests=flags_obj.num_interests,
                                CL_thr=flags_obj.CL_thr,
                                k_size=flags_obj.k_size,
                                alpha=flags_obj.alpha,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path
                    )
        model = Model(hparams, input_creator, seed=None)
        
    print(hparams)

    return model, model_path, summary_path

def main(argv):
    
    flags_obj = FLAGS
    os.environ["CUDA_VISIBLE_DEVICES"]=f'{flags_obj.gpu_id}'

    if flags_obj.model in ['model']:
        exp_name = f'{Decimal(flags_obj.learning_rate):.0E}_{Decimal(flags_obj.embed_l2):.0E}_{Decimal(flags_obj.layer_l2):.0E}_alpha{flags_obj.alpha}_recent{flags_obj.recent_k}_interest{flags_obj.num_interests}_thr{flags_obj.CL_thr}_ksize{flags_obj.k_size}fatigue_weight_{flags_obj.fatigue_weight}_fatigue_emb_{flags_obj.fatigue_emb}_{flags_obj.name}'
    elif flags_obj.model in ['slirec', 'gru4rec', 'dien', 'din', 'dfn']:
        exp_name = f'{Decimal(flags_obj.learning_rate):.0E}_{Decimal(flags_obj.embed_l2):.0E}_{Decimal(flags_obj.layer_l2):.0E}_{flags_obj.name}'
    elif flags_obj.model in ['sasrec']:
        exp_name = f'{Decimal(flags_obj.learning_rate):.0E}_{Decimal(flags_obj.embed_l2):.0E}_{Decimal(flags_obj.layer_l2):.0E}_blocks{flags_obj.num_blocks}_heads{flags_obj.attention_num_heads}_{flags_obj.name}'
    elif flags_obj.model in ['sum', 'mimn', 'comi']:
        exp_name = f'{Decimal(flags_obj.learning_rate):.0E}_{Decimal(flags_obj.embed_l2):.0E}_{Decimal(flags_obj.layer_l2):.0E}_{flags_obj.slots}_{flags_obj.name}'
    elif flags_obj.model in ['mgnm']:
        exp_name = f'{Decimal(flags_obj.learning_rate):.0E}_{Decimal(flags_obj.embed_l2):.0E}_{Decimal(flags_obj.layer_l2):.0E}_{flags_obj.num_levels}_{flags_obj.slots}_{flags_obj.name}'
    elif flags_obj.model in ['clsr']:
        exp_name = f'{Decimal(flags_obj.learning_rate):.0E}_{Decimal(flags_obj.embed_l2):.0E}_{Decimal(flags_obj.layer_l2):.0E}_CL{flags_obj.contrastive_loss_weight}_{flags_obj.contrastive_length_threshold}_{flags_obj.name}'
    elif flags_obj.model in ['caser']:
        exp_name = f'{Decimal(flags_obj.learning_rate):.0E}_{Decimal(flags_obj.embed_l2):.0E}_{Decimal(flags_obj.layer_l2):.0E}_{flags_obj.L}_{flags_obj.nv}_{flags_obj.nh}_{flags_obj.name}'


    model, model_path, _ = get_model(flags_obj, exp_name)
    
    if flags_obj.dataset in ['taobao']:
        train_file = os.path.join(flags_obj.data_path, f'{flags_obj.dataset}/train_data')
        valid_file = os.path.join(flags_obj.data_path, f'{flags_obj.dataset}/valid_data')
        test_file = os.path.join(flags_obj.data_path, f'{flags_obj.dataset}/test_data')
    else:
        raise ValueError('No this dataset')
    
    if flags_obj.only_test:
        ckpt_path = os.path.join(model_path,'best_model')
        model.load_model(ckpt_path)
        res = model.run_eval(test_file, num_ngs=flags_obj.test_num_ngs)
        print(res)
        return

    with Timer() as train_time:
        model = model.fit(train_file, valid_file, valid_num_ngs=flags_obj.valid_num_ngs) 

    print('Time cost for training is {0:.2f} mins'.format(train_time.interval/60.0))

    ckpt_path = os.path.join(model_path,'best_model')
    model.load_model(ckpt_path)
    res = model.run_eval(test_file, num_ngs=flags_obj.test_num_ngs)
    print(res)

if __name__ == "__main__":
    
    app.run(main)