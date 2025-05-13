import tensorflow as tf
import numpy as np
import torch
import os
# pandas is no longer needed
from tqdm import tqdm # Keep tqdm for progress

from ogb.utils import smiles2graph
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
# Assuming ASOptimizer class is accessible
from main import ASOptimizer # Or from libml.models import EGT_Model

# Import types from the typing module for compatibility with Python < 3.9
from typing import List, Tuple, Optional

# --- Configuration (Match parameters used during training) ---
NODE_DIM = 64           # From main.py default flags
EDGE_DIM = 64           # From main.py default flags
MAX_LENGTH = 516        # From main.py default flags
MODEL_HEIGHT = 12       # From main.py default flags
NUM_HEAD = 32           # From main.py default flags
NUM_VNODE = 8           # From main.py default flags
MASK_VALUE = -1.0       # Default mask value used in Masking layers
CHECKPOINT_PATH = './checkpoints/training_checkpoints/best_checkpoint' # Path to saved weights

# --- Input/Output File Configuration ---
INPUT_TEXT_PATH = '../asogym/smiles.txt'
OUTPUT_TEXT_PATH = './scores.txt' # Output file name

# --- Chunking Configuration ---
CHUNK_SIZE = 16 # Process 1024 SMILES strings per batch (adjust based on memory)

# --- Global variables for lazy loading ---
_model = None
_atom_encoder = None
_bond_encoder = None

# --- Model Loading and Preprocessing Functions (Unchanged from previous version) ---

def _load_model_and_encoders():
    """Loads the Keras model and OGB encoders once."""
    global _model, _atom_encoder, _bond_encoder
    if _model is None:
        print("Loading model and encoders...")
        model_builder = ASOptimizer()
        _model = model_builder.EGT_Backbone(
            node_dim=NODE_DIM,
            edge_dim=EDGE_DIM,
            model_height=MODEL_HEIGHT,
            num_heads=NUM_HEAD,
            num_virtual_nodes=NUM_VNODE,
            max_length=MAX_LENGTH
        )
        if os.path.exists(CHECKPOINT_PATH + '.index'):
            _model.load_weights(CHECKPOINT_PATH).expect_partial()
            print(f"Model weights loaded from {CHECKPOINT_PATH}")
        else:
             raise FileNotFoundError(
                f"Checkpoint not found at {CHECKPOINT_PATH}. "
                "Please ensure 'best_checkpoint.index' and corresponding "
                "'.data-...' files exist in ./checkpoints/training_checkpoints/"
            )

    if _atom_encoder is None:
        _atom_encoder = AtomEncoder(emb_dim=NODE_DIM)
    if _bond_encoder is None:
        _bond_encoder = BondEncoder(emb_dim=EDGE_DIM)

    return _model, _atom_encoder, _bond_encoder

def _preprocess_smiles_batch(smiles_list: List[str], atom_encoder, bond_encoder) -> Tuple[Optional[tf.Tensor], Optional[tf.Tensor], Optional[tf.Tensor], List[Optional[float]], List[int]]:
    """
    Converts a list of SMILES strings into batched, model-compatible TF tensors.
    Handles errors gracefully by skipping invalid SMILES.
    Expects input smiles_list where empty strings represent originally empty lines.
    """
    valid_node_arrays = []
    valid_adj_arrays = []
    valid_edge_arrays = []
    valid_indices = []
    # Initialize template with None, matching the input list length
    results_template: List[Optional[float]] = [None] * len(smiles_list)

    for i, smiles_string in enumerate(smiles_list):
        # Treat empty strings (from originally empty or failed lines) as invalid for processing
        if not isinstance(smiles_string, str) or not smiles_string:
            continue # Skip non-strings and empty strings

        try:
            graph_data = smiles2graph(smiles_string)

            # --- Feature Encoding ---
            atom_feats_torch = atom_encoder(torch.LongTensor(graph_data['node_feat']))
            bond_feats_torch = bond_encoder(torch.LongTensor(graph_data['edge_feat']))
            atom_feats = atom_feats_torch.detach().numpy()
            bond_feats = bond_feats_torch.detach().numpy()
            num_atoms = atom_feats.shape[0]

            if num_atoms > MAX_LENGTH:
                # print(f"Warning: Skipping SMILES at index {i} (relative). Too long.")
                continue

            # --- NumPy Array Creation ---
            node_features_np = MASK_VALUE * np.ones((MAX_LENGTH, NODE_DIM), dtype=np.float32)
            node_features_np[:num_atoms] = atom_feats

            adj_matrix_np = np.zeros((MAX_LENGTH, MAX_LENGTH), dtype=np.float32)
            edge_index = graph_data['edge_index']
            if edge_index.shape[1] > 0:
                adj_matrix_np[edge_index[0], edge_index[1]] = 1
                adj_matrix_np[edge_index[1], edge_index[0]] = 1
            adj_matrix_np[np.arange(num_atoms), np.arange(num_atoms)] = 1 # Self-loops

            edge_features_np = MASK_VALUE * np.ones((MAX_LENGTH, MAX_LENGTH, EDGE_DIM), dtype=np.float32)
            if edge_index.shape[1] > 0:
                 for bond_idx in range(edge_index.shape[1]):
                     if bond_idx < len(bond_feats): # Check bounds
                        u, v = edge_index[0, bond_idx], edge_index[1, bond_idx]
                        edge_features_np[u, v] = bond_feats[bond_idx]
                        edge_features_np[v, u] = bond_feats[bond_idx]

            # Store valid arrays and *relative* index within the chunk
            valid_node_arrays.append(node_features_np)
            valid_adj_arrays.append(adj_matrix_np)
            valid_edge_arrays.append(edge_features_np)
            valid_indices.append(i)

        except ValueError: # Catch OGB error for invalid SMILES format
             # print(f"Warning: Skipping SMILES at relative index {i} due to invalid format.")
             continue
        except Exception: # Catch other unexpected errors during graph creation/padding
            # print(f"Warning: Skipping SMILES at relative index {i} due to unexpected preprocessing error: {e}")
            continue

    # --- Stack and Convert to Tensor ---
    batched_node_tf: Optional[tf.Tensor] = None
    batched_adj_tf: Optional[tf.Tensor] = None
    batched_edge_tf: Optional[tf.Tensor] = None
    if valid_indices:
        try:
            batched_node_np = np.stack(valid_node_arrays, axis=0)
            batched_adj_np = np.stack(valid_adj_arrays, axis=0)
            batched_edge_np = np.stack(valid_edge_arrays, axis=0)

            batched_node_tf = tf.constant(batched_node_np, dtype=tf.float32)
            batched_adj_tf = tf.constant(batched_adj_np, dtype=tf.float32)
            batched_edge_tf = tf.constant(batched_edge_np, dtype=tf.float32)
        except Exception as stack_err:
            print(f"Error during stacking/tensor conversion for a chunk: {stack_err}. Invalidating chunk.")
            batched_node_tf, batched_adj_tf, batched_edge_tf = None, None, None
            valid_indices = [] # Mark chunk as having no valid indices if stacking fails

    # Return tensors (or None), the template list, and the list of valid *relative* indices
    return batched_node_tf, batched_adj_tf, batched_edge_tf, results_template, valid_indices

def get_scores_for_smiles_batch(smiles_list: List[str]) -> List[Optional[float]]:
    """
    Maps a list of SMILES strings (a chunk) to their prediction scores.
    Handles empty strings in the input list by returning None at that position.
    """
    if not smiles_list:
        return []

    # Initialize results with None for all inputs (including empty strings)
    results = [None] * len(smiles_list)

    # Filter out empty strings and track their original indices
    valid_smiles_with_indices = [(idx, smi) for idx, smi in enumerate(smiles_list) if smi.strip()]

    if not valid_smiles_with_indices:
        return results  # all None, already initialized

    valid_indices, valid_smiles = zip(*valid_smiles_with_indices)

    print(len(valid_indices), len(valid_smiles))

    # Preload model/encoders if needed
    model, atom_encoder, bond_encoder = _load_model_and_encoders()

    # Preprocess only valid SMILES strings
    node_tf, adj_tf, edge_tf, _, relative_indices = _preprocess_smiles_batch(
        list(valid_smiles), atom_encoder, bond_encoder
    )

    # Only run inference if there were valid SMILES and tensors were created
    if relative_indices and node_tf is not None and adj_tf is not None and edge_tf is not None:
        try:
            predictions = model([node_tf, adj_tf, edge_tf], training=False)
            batch_scores_np = predictions[0].numpy()

            # Map scores back to original indices in the full list
            for batch_idx, rel_idx in enumerate(relative_indices):
                if batch_idx < batch_scores_np.shape[0]:
                    original_idx = valid_indices[rel_idx]
                    results[original_idx] = float(batch_scores_np[batch_idx, 0])

        except Exception as inference_err:
            print(f"\nError during model inference: {inference_err}. Assigning None to all valid entries.")
            for rel_idx in relative_indices:
                original_idx = valid_indices[rel_idx]
                results[original_idx] = None

    return results

# --- Main Execution Logic ---
if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR') # Reduce TensorFlow verbosity

    # Ensure model is loaded *before* the loop starts
    _load_model_and_encoders()
    print("-" * 30)

    print(f"Processing SMILES from: {INPUT_TEXT_PATH}")
    print(f"Writing scores to: {OUTPUT_TEXT_PATH}")
    print(f"Using chunk size: {CHUNK_SIZE}")

    processed_lines = 0
    try:
        # Open both input and output files
        with open(INPUT_TEXT_PATH, 'r') as infile, open(OUTPUT_TEXT_PATH, 'w') as outfile:
            # Use tqdm to monitor line processing
            # Need to estimate total lines for tqdm, which requires reading the file once or seeking.
            # Simpler approach: Just use tqdm without total if file size is unknown/large.
            # Or, count lines first (might be slow for huge files)
            # total_lines = sum(1 for line in open(INPUT_TEXT_PATH, 'r')) # Example line count
            # print(f"Estimated total lines: {total_lines}")

            chunk_lines: List[str] = []
            tqdm_iterator = tqdm(infile, desc="Processing Lines", unit=" lines")


            for line in tqdm_iterator:
                # Strip newline/whitespace and add to the current chunk
                chunk_lines.append(line.strip())
                processed_lines += 1

                # When chunk is full or it's the last line (though tqdm handles end)
                if len(chunk_lines) == CHUNK_SIZE:
                    # Process the chunk
                    chunk_scores = get_scores_for_smiles_batch(chunk_lines)

                    # Write results for this chunk to the output file
                    for score in chunk_scores:
                        if score is None:
                            outfile.write("\n") # Write empty line for None/Error/Empty Input
                        else:
                            outfile.write(f"{score:.8f}\n") # Write formatted score

                    # Clear the chunk for the next iteration
                    chunk_lines = []

            # Process any remaining lines in the last (potentially smaller) chunk
            if chunk_lines:
                chunk_scores = get_scores_for_smiles_batch(chunk_lines)
                for score in chunk_scores:
                    if score is None:
                        outfile.write("\n")
                    else:
                        outfile.write(f"{score:.8f}\n")

    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_TEXT_PATH}")
        exit()
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        exit()

    print("-" * 30)
    print(f"Processing complete. Total lines processed: {processed_lines}")
    print(f"Scores written to {OUTPUT_TEXT_PATH}")