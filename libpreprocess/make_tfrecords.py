import numpy as np
import pandas as pd
import tensorflow as tf
import os
import shutil
from absl import flags, app
from tqdm import tqdm
from multiprocessing import Pool
from ogb.utils import smiles2graph
import random
import torch
import argparse
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data_type', type=str, help='Your name')

FLAGS = flags.FLAGS
args = parser.parse_args()
data_type = args.data_type


OUTPUT_PATH = './data/{}'.format(data_type)
NUM_WORKERS = 16
n_node_features = 64

seed=42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def generate_edge_features(molecule_id, n_node_features, data_dict, edge_dict):
    """
    Generate edge features for a given molecule.
    """
    edge_indices = np.array(data_dict[molecule_id]['edge_index'])
    edge_values = edge_dict[molecule_id]

    edge_i_repeat = np.repeat(edge_indices, n_node_features, axis=1)
    edge_i_tile = np.tile(np.arange(n_node_features), edge_indices.shape[1])
    edge_merge = np.vstack((edge_i_repeat, edge_i_tile))

    return edge_indices, edge_values, edge_merge


def add_self_loops(edge_indices, n_length):
    """
    Add self-loops to edge indices.
    """
    self_loops = np.argwhere(np.eye(n_length) == 1)
    return np.hstack((edge_indices, self_loops.T))

def process_row_screening(row, n_node, n_node_features, atom_dict, length, data_dict, edge_dict):
    """
    Processes a single row for screening mode.
    """
    screening_id = row['screening_id']
    inhibition = row['Inhibition(%)']

    # Initialize node features
    X_front = -1 * np.ones((n_node, n_node_features))
    X_front[:atom_dict[screening_id].shape[0]] = atom_dict[screening_id]

    # Retrieve lengths and inhibition values
    n_length = length[screening_id]
    pairs_id = screening_id
    inhibition_value = inhibition

    # Generate edge features
    front_feat_i, front_feat_v, front_feat_merge = generate_edge_features(
        screening_id, n_node_features, data_dict, edge_dict)

    # Add self-loops
    front_feat_i = add_self_loops(front_feat_i, n_length)

    return X_front, front_feat_i, front_feat_v, front_feat_merge, pairs_id, inhibition_value


def process_row_pairs(row, n_node, n_node_features, atom_dict, length, data_dict, edge_dict):
    """
    Processes a single row for general mode.
    """
    front_id = row['front_id']
    back_id = row['back_id']
    label = row['labels']

    # Initialize node features
    X_front = -1 * np.ones((n_node, n_node_features))
    X_back = -1 * np.ones((n_node, n_node_features))
    X_front[:atom_dict[front_id].shape[0]] = atom_dict[front_id]
    X_back[:atom_dict[back_id].shape[0]] = atom_dict[back_id]

    # Retrieve lengths
    n_length_front = length[front_id]
    n_length_back = length[back_id]
    pairs_id = np.array([front_id, back_id])

    # Generate edge features
    front_feat_i, front_feat_v, front_feat_merge = generate_edge_features(
        front_id, n_node_features, data_dict, edge_dict)
    back_feat_i, back_feat_v, back_feat_merge = generate_edge_features(
        back_id, n_node_features, data_dict, edge_dict)

    # Add self-loops
    front_feat_i = add_self_loops(front_feat_i, n_length_front)
    back_feat_i = add_self_loops(back_feat_i, n_length_back)

    return X_front, front_feat_i, front_feat_v, front_feat_merge, X_back, back_feat_i, back_feat_v, back_feat_merge, pairs_id, label

def processing(df,n_node, data_dict, atom_dict,edge_dict, length, mode, chunk_size):
    
    pool = Pool(NUM_WORKERS)
    chunk_offsets = _generate_chunks(chunk_size, len(df))

    try:
        results  = pool.map_async(_process_chunk, [(cidx, begin, end, df, mode, n_node, length, data_dict, atom_dict,edge_dict)
                                                for cidx, (begin, end) in enumerate(chunk_offsets)]).get(999999999)
        pool.close()
        pool.join()

    except KeyboardInterrupt:

        print("Processing interrupted. Terminating...")
        pool.terminate()
        pool.join()
        raise

    total_samples = sum(results)
    print(f"Processed {mode} data. Total samples: {total_samples}")


def _generate_chunks(chunk_size, total_length):

    return [(start, min(start + chunk_size, total_length))
                for start in range(0, total_length, chunk_size)]


def _process_chunk(data):

    cidx, begin, end, dataframe, mode, n_node ,length, data_dict, atom_dict, edge_dict = data

    with tqdm(total=end - begin, desc=f"Processing Chunk {cidx}") as pbar:

        output_path = os.path.join(OUTPUT_PATH, 'tfrecords', f"{mode}.chunk{cidx}.tfrecord")
        writer = tf.io.TFRecordWriter(output_path)

        df_partial = dataframe.iloc[begin:end]
        nums_pos = 0

        if mode =='screening':

            for _, row in df_partial.iterrows():

                X_front, front_feat_i, front_feat_v, front_feat_merge, pairs_id, inhibition_value = process_row_screening(
                row, n_node, n_node_features, atom_dict, length, data_dict, edge_dict)
                
                nums_pos += 1

                feat = {'front_feat': _bytes_feature(X_front.astype(np.float32).reshape([-1]).tobytes()),
                        'front_feat_merge': _bytes_feature(front_feat_merge.astype(np.int64).reshape([-1]).tobytes()),
                        'front_feat_i': _bytes_feature(front_feat_i.astype(np.int64).reshape([-1]).tobytes()),
                        'front_feat_v': _bytes_feature(front_feat_v.astype(np.float32).reshape([-1]).tobytes()),
                        'pairs_id': _bytes_feature(pairs_id.astype(np.int64).reshape([-1]).tobytes()),
                        'Inhibition': _bytes_feature(inhibition_value.astype(np.float32).reshape([-1]).tobytes()),
                        }

                record = tf.train.Example(features=tf.train.Features(feature=feat))
                writer.write(record.SerializeToString())
                pbar.update(1)


        else:
            for _, row in df_partial.iterrows():

                X_front,front_feat_i, front_feat_v, front_feat_merge, X_back,  \
                back_feat_i, back_feat_v, back_feat_merge, pairs_id, label = \
                process_row_pairs(row,n_node,n_node_features,atom_dict,length,data_dict,edge_dict)

                nums_pos += 1

                feat = {'front_feat': _bytes_feature(X_front.astype(np.float32).reshape([-1]).tobytes()),
                        'back_feat': _bytes_feature(X_back.astype(np.float32).reshape([-1]).tobytes()),
                        'front_feat_merge': _bytes_feature(front_feat_merge.astype(np.int64).reshape([-1]).tobytes()),
                        'front_feat_i': _bytes_feature(front_feat_i.astype(np.int64).reshape([-1]).tobytes()),
                        'front_feat_v': _bytes_feature(front_feat_v.astype(np.float32).reshape([-1]).tobytes()),
                        'back_feat_merge': _bytes_feature(back_feat_merge.astype(np.int64).reshape([-1]).tobytes()),
                        'back_feat_i': _bytes_feature(back_feat_i.astype(np.int64).reshape([-1]).tobytes()),
                        'back_feat_v': _bytes_feature(back_feat_v.astype(np.float32).reshape([-1]).tobytes()),
                        'pairs_id': _bytes_feature(pairs_id.reshape([-1]).tobytes()),
                        'label': _int64_feature(int(label)),
                        }

                record = tf.train.Example(features=tf.train.Features(feature=feat))
                writer.write(record.SerializeToString())

                pbar.update(1)

    writer.close()
    return nums_pos

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def process_training_data():
    # Load datasets
    df_exps = pd.read_csv('dataset/experiments_with_smiles.csv')
    train_data = pd.read_csv('dataset/train_set_id.csv')
    test_data = pd.read_csv('dataset/test_set_id.csv')

    print(f"# of train pairs: {len(train_data)}")
    print(f"# of test pairs: {len(test_data)}")

    # Collect unique IDs
    isis_list = set(
        train_data['front_id']).union(
        train_data['back_id'], 
        test_data['front_id'], 
        test_data['back_id'])

    # Generate graph data
    data_dict = {ii: smiles2graph(df_exps[df_exps['ISIS'] == ii]['Smiles'].values[0]) for ii in isis_list}

    # Encode atom and bond features
    atom_dict, edge_dict, n_atom_dict = encode_graph_data(data_dict)
    n_length = np.max(list(n_atom_dict.values()))
    n_node = 516  # Max nodes configuration

    print(f"# of max length: {n_length}")
    print(f"# of nodes: {n_node}")

    # Process training and test data
    processing(train_data, n_node, data_dict, atom_dict, edge_dict, n_atom_dict, 'training', chunk_size=(len(train_data) // NUM_WORKERS) + 1)
    processing(test_data, n_node, data_dict, atom_dict, edge_dict, n_atom_dict, 'test', chunk_size=(len(test_data) // NUM_WORKERS) + 1)

def process_screening_data():
    # Load screening data
    df_hif1a_exp = pd.read_csv('dataset/HIF1A/experiments.csv', encoding='cp949')
    df_hif1a_smile = pd.read_csv('dataset/HIF1A/smiles.csv', encoding='cp949')

    roche_data = pd.DataFrame({
        'screening_id': df_hif1a_exp['ISIS'],
        'Inhibition(%)': df_hif1a_exp['Inhibition(%)']
    })

    print(f"# of roche pairs: {len(roche_data)}")

    # Generate graph data
    isis_list_roche = df_hif1a_exp['ISIS']
    data_dict = {ii: smiles2graph(df_hif1a_smile[df_hif1a_smile['ISIS'] == ii]['smiles'].values[0]) for ii in isis_list_roche}

    # Encode atom and bond features
    atom_dict, edge_dict, n_atom_dict = encode_graph_data(data_dict)
    n_length = np.max(list(n_atom_dict.values()))
    n_node = 516  # Max nodes configuration

    # Process screening data
    processing(roche_data, n_node, data_dict, atom_dict, edge_dict, n_atom_dict, 'screening', chunk_size=(len(roche_data) // NUM_WORKERS) + 1)

def encode_graph_data(data_dict):
    atom_encoder = AtomEncoder(emb_dim=n_node_features)
    bond_encoder = BondEncoder(emb_dim=n_node_features)
    atom_dict, edge_dict, n_atom_dict = {}, {}, {}

    for ii in data_dict:
        atom_feats = atom_encoder(torch.LongTensor(data_dict[ii]['node_feat'])).detach().numpy()
        bond_feats = bond_encoder(torch.LongTensor(data_dict[ii]['edge_feat'])).detach().numpy()
        atom_dict[ii] = atom_feats
        edge_dict[ii] = bond_feats
        n_atom_dict[ii] = len(atom_feats)

    return atom_dict, edge_dict, n_atom_dict

def main(argv):
    del argv

    if not os.path.exists(OUTPUT_PATH): 
        os.mkdir(OUTPUT_PATH)

    tfrecords_path = os.path.join(OUTPUT_PATH, 'tfrecords')
    if os.path.exists(tfrecords_path):
        shutil.rmtree(tfrecords_path)
    os.mkdir(tfrecords_path)

    # Data processing for training
    if data_type == 'training':
        process_training_data()

    # Data processing for screening
    elif data_type == 'screening':
        process_screening_data()

if __name__ == '__main__':
    flags.DEFINE_integer('random_seed', 0, 'Seed')
    flags.DEFINE_string('data_type', 'training', 'training or screening')

    app.run(main)
