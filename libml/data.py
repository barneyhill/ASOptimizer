import tensorflow as tf
import glob
import numpy as np
from libml import utils
from absl import flags
import pickle
import random
import torch

flags.DEFINE_integer('random_seed', 0, 'Seed.')
flags.DEFINE_integer('para_parse', 4, 'Parallel parsing.')
flags.DEFINE_integer('para_augment', 4, 'Parallel augmentation.')
flags.DEFINE_integer('shuffle', 16384, 'Size of dataset shuffling.')
flags.DEFINE_bool('whiten', False, 'Whether to anormalize images.')
FLAGS = flags.FLAGS

seed=42
tf.random.set_seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

def record_parse(serialized_example):

    n_nodes = FLAGS.max_length
    feat_dim = FLAGS.node_dim

    features = tf.io.parse_single_example(
        serialized_example,
        features={'front_feat'       : tf.io.FixedLenFeature([], tf.string),
                  'back_feat'      : tf.io.FixedLenFeature([], tf.string),
                  'front_feat_v': tf.io.FixedLenFeature([], tf.string),
                  'back_feat_v': tf.io.FixedLenFeature([], tf.string),
                  'front_feat_i': tf.io.FixedLenFeature([], tf.string),
                  'back_feat_i': tf.io.FixedLenFeature([], tf.string),
                  'front_feat_merge': tf.io.FixedLenFeature([], tf.string),
                  'back_feat_merge': tf.io.FixedLenFeature([], tf.string),
                  'pairs_id': tf.io.FixedLenFeature([], tf.string),
                  'label'      : tf.io.FixedLenFeature([], tf.int64)
                  })

    front_feat = tf.io.decode_raw(features['front_feat'], out_type='float32')
    front_feat = tf.reshape(front_feat, [n_nodes, feat_dim])

    back_feat = tf.io.decode_raw(features['back_feat'], out_type='float32')
    back_feat = tf.reshape(back_feat, [n_nodes, feat_dim])

    front_feat_v = tf.io.decode_raw(features['front_feat_v'], out_type='float32')
    back_feat_v = tf.io.decode_raw(features['back_feat_v'], out_type='float32')

    front_feat_merge = tf.io.decode_raw(features['front_feat_merge'], out_type='int64')
    front_feat_merge = tf.reshape(front_feat_merge, [3, -1])

    back_feat_merge = tf.io.decode_raw(features['back_feat_merge'], out_type='int64')
    back_feat_merge = tf.reshape(back_feat_merge, [3, -1])

    front_feat_i = tf.io.decode_raw(features['front_feat_i'], out_type='int64')
    front_feat_i = tf.reshape(front_feat_i, [2, -1])

    back_feat_i = tf.io.decode_raw(features['back_feat_i'], out_type='int64')
    back_feat_i = tf.reshape(back_feat_i, [2, -1])

    pairs_id = tf.io.decode_raw(features['pairs_id'], out_type='int32')

    front_e_feat = tf.sparse.SparseTensor(indices=tf.transpose(front_feat_merge), values=front_feat_v+1, dense_shape=[FLAGS.max_length, FLAGS.max_length, FLAGS.edge_dim])
    back_e_feat = tf.sparse.SparseTensor(indices=tf.transpose(back_feat_merge), values=back_feat_v+1,dense_shape=[FLAGS.max_length, FLAGS.max_length, FLAGS.edge_dim])

    front_e_feat = tf.sparse.reorder(front_e_feat)
    front_e_feat = tf.sparse.to_dense(front_e_feat, default_value=None, validate_indices=True, name=None)
    front_e_feat -= 1

    back_e_feat = tf.sparse.reorder(back_e_feat)
    back_e_feat = tf.sparse.to_dense(back_e_feat, default_value=None, validate_indices=True, name=None)
    back_e_feat -= 1

    front_adj = tf.sparse.SparseTensor(indices=tf.transpose(front_feat_i), values=tf.ones_like(front_feat_i)[0], dense_shape=[FLAGS.max_length, FLAGS.max_length])
    back_adj = tf.sparse.SparseTensor(indices=tf.transpose(back_feat_i), values=tf.ones_like(back_feat_i)[0],dense_shape=[FLAGS.max_length, FLAGS.max_length])


    front_adj = tf.sparse.reorder(front_adj)
    front_adj = tf.sparse.to_dense(front_adj, default_value=None, validate_indices=True, name=None)

    back_adj = tf.sparse.reorder(back_adj)
    back_adj = tf.sparse.to_dense(back_adj, default_value=None, validate_indices=True, name=None)

    labels = features['label']#tf.reshape(features['label'], [1])
    return dict(front_feat=front_feat,back_feat=back_feat,front_adj=front_adj,back_adj=back_adj,front_e_feat=front_e_feat,back_e_feat=back_e_feat,pairs_id=pairs_id, labels=labels)

def record_screen_parse(serialized_example):

    n_nodes = FLAGS.max_length
    feat_dim = FLAGS.node_dim

    features = tf.io.parse_single_example(
        serialized_example,
        features={'front_feat'       : tf.io.FixedLenFeature([], tf.string),
                  'front_feat_merge': tf.io.FixedLenFeature([], tf.string),
                  'front_feat_i': tf.io.FixedLenFeature([], tf.string),
                  'front_feat_v': tf.io.FixedLenFeature([], tf.string),
                  'pairs_id': tf.io.FixedLenFeature([], tf.string),
                  'Inhibition': tf.io.FixedLenFeature([], tf.string), #tf.int64
                  })

    front_feat = tf.io.decode_raw(features['front_feat'], out_type='float32')
    front_feat = tf.reshape(front_feat, [n_nodes, feat_dim])


    front_feat_v = tf.io.decode_raw(features['front_feat_v'], out_type='float32')

    front_feat_merge = tf.io.decode_raw(features['front_feat_merge'], out_type='int64')
    front_feat_merge = tf.reshape(front_feat_merge, [3, -1])

    front_feat_i = tf.io.decode_raw(features['front_feat_i'], out_type='int64')
    front_feat_i = tf.reshape(front_feat_i, [2, -1])

    front_e_feat = tf.sparse.SparseTensor(indices=tf.transpose(front_feat_merge), values=front_feat_v+1, dense_shape=[FLAGS.max_length, FLAGS.max_length, FLAGS.edge_dim])
    front_e_feat = tf.sparse.reorder(front_e_feat)
    front_e_feat = tf.sparse.to_dense(front_e_feat, default_value=None, validate_indices=True, name=None)
    front_e_feat -= 1

    front_adj = tf.sparse.SparseTensor(indices=tf.transpose(front_feat_i), values=tf.ones_like(front_feat_i)[0], dense_shape=[FLAGS.max_length, FLAGS.max_length])

    front_adj = tf.sparse.reorder(front_adj)
    front_adj = tf.sparse.to_dense(front_adj, default_value=None, validate_indices=True, name=None)

    pairs_id = tf.io.decode_raw(features['pairs_id'],out_type='int64') #int32
    # paris_id = tf.reshape(pairs_id,[-1])

    Inhibition = tf.io.decode_raw(features['Inhibition'],out_type='float32')
    # pairs_id = features['pairs_id']

    return dict(front_feat=front_feat,front_adj=front_adj,front_e_feat=front_e_feat,pairs_id=pairs_id,Inhibition=Inhibition)


def default_parse(dataset: tf.data.Dataset, parse_fn=record_parse) -> tf.data.Dataset:

    para = 4 * max(1, len(utils.get_available_gpus())) * 4
    return dataset.map(parse_fn, num_parallel_calls=para)

def screen_parse(dataset: tf.data.Dataset, parse_fn=record_screen_parse) -> tf.data.Dataset:

    para = 4 * max(1, len(utils.get_available_gpus())) * 4
    return dataset.map(parse_fn, num_parallel_calls=para)


def dataset(filenames: list) -> tf.data.Dataset:
    filenames = sorted(sum([glob.glob(x) for x in filenames], []))
    if not filenames:
        raise ValueError('Empty dataset, did you mount gcsfuse bucket?')

    return tf.data.TFRecordDataset(filenames)

class DataSet:
    def __init__(self, name, train,  test, screen):
        self.name   = name
        self.train  = train
        self.test   = test
        self.screen   = screen

    @classmethod
    def creator(cls, name, augment, augment_valid, augment_screen, parse_fn=default_parse, parse_fn_screen=screen_parse):
        fn = lambda x: x.repeat()
        def create():
            DATA_DIR   = './data/training/tfrecords/'
            TEST_DIR   = './data/training/tfrecords/'
            RANK_DIR = './data/screening/tfrecords/'

            # para = max(1, len(utils.get_available_gpus())) * FLAGS.para_augment

            para = max(1, len(utils.get_available_gpus())) * 4
            TRAIN_DATA      = [DATA_DIR + 'training.chunk*.tfrecord']
            TEST_DATA       = [DATA_DIR + 'test.chunk*.tfrecord']
            SCREEN_DATA       = [RANK_DIR + 'screening.chunk*.tfrecord']

            train_data      = parse_fn(dataset(TRAIN_DATA))
            test_data      = parse_fn(dataset(TEST_DATA))
            screen_data      = parse_fn_screen(dataset(SCREEN_DATA))

            b1 = train_data.shuffle(8, reshuffle_each_iteration=True).batch(FLAGS.batch)
            return cls(name,
                       train = b1.map(augment, para),
                       test = test_data.batch(FLAGS.batch).map(augment_valid, para),
                       screen = screen_data.batch(FLAGS.batch).map(augment_screen, para))
        return name, create

augment_train = lambda x: ({'front_feat': x['front_feat'],
                            'back_feat' : x['back_feat'],
                            'front_adj': x['front_adj'],
                            'back_adj': x['back_adj'],
                            'front_e_feat': x['front_e_feat'],
                            'back_e_feat': x['back_e_feat']},
                           {'labels'    : x['labels']})

augment_valid = lambda x: ({'front_feat': x['front_feat'],
                            'back_feat' : x['back_feat'],
                            'front_adj': x['front_adj'],
                            'back_adj': x['back_adj'],
                            'front_e_feat': x['front_e_feat'],
                            'back_e_feat': x['back_e_feat']},
                           {'labels'    : x['labels'],'pairs_id': x['pairs_id']},)

augment_screen = lambda x: ({'front_feat': x['front_feat'],
                            'front_adj' : x['front_adj'],
                            'front_e_feat': x['front_e_feat'],
                            'pairs_id': x['pairs_id'],
                            'Inhibition': x['Inhibition']})

DATASETS = {}
DATASETS.update([DataSet.creator('chemical_engineering', augment_train, augment_valid, augment_screen)])
