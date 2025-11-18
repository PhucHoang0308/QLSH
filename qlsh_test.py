import pennylane as qml
import numpy as np
import sys
from functools import partial
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import time
from datetime import timedelta
import pandas as pd
import numpy as np
import os
from pathlib import Path

def Quantum_Hamming_Distance(dataset, query_list, length, length_dist, P, k, query_idx=0, save_circuit=True):

    """
    dataset: List of size N of the P partitions lists of samples
    query_list: List of size P of query paritions
    length: Length of 1 partition
    P: Number of partition per sample
    query_idx: Index of current query (for saving circuit diagram)
    save_circuit: Whether to save circuit diagram
    """

    index_reg_qubits = int(np.ceil(np.log2(len(dataset))))
    number_of_qubits = length*2 + length_dist + index_reg_qubits + 1

    index_reg = list(range(index_reg_qubits))
    data_reg = list(range(index_reg_qubits, index_reg_qubits+length))
    query_reg = list(range(index_reg_qubits+length, index_reg_qubits+length+length))
    distance_reg = list(range(index_reg_qubits+length+length, index_reg_qubits+length+length+length_dist))
    ancilla_qubit = index_reg_qubits+length+length+length_dist
    shots = max(20, int(np.ceil(len(dataset)*np.log2(max(2, len(dataset))))))  # Ensure minimum 20 shots



    def Hamming_circuit(data_reg_idx, query_reg_idx, length):
        for n in range(length):
            qml.CNOT(wires=[query_reg_idx[n], data_reg_idx[n]])

    def Prepare_index_register(index_reg_idx, data_size):
        index_state = [1/np.sqrt(data_size)]*data_size+[0]*(2**(len(index_reg_idx))-data_size)
        qml.StatePrep(index_state,index_reg_idx)

    def InC_circuit(query_reg_idx, distance_reg_idx, ancilla_qubit_idx, length):
        for n in range(length):
            qml.PauliX(ancilla_qubit)
            for m in range(length_dist-1):
                qml.Toffoli(wires=[query_reg_idx[n], ancilla_qubit_idx, distance_reg_idx[m]])
                control_wires = [query_reg_idx[n],distance_reg_idx[0]] + distance_reg_idx[1:m+1] if m !=0 else [query_reg_idx[n],distance_reg_idx[0]]
                control_values = [1] + [0]*(m) + [1] if m !=0 else [1,1]
                qml.MultiControlledX(wires=control_wires+[ancilla_qubit_idx], control_values=control_values)
            qml.Toffoli(wires=[query_reg_idx[n], ancilla_qubit_idx, distance_reg_idx[-1]])
            qml.X(wires=ancilla_qubit_idx)
            qml.MultiControlledX(wires=[query_reg_idx[n]]+distance_reg_idx[0:length_dist-1]+[ancilla_qubit_idx], control_values=[1]+[0]*(length_dist-1))
            qml.measure(ancilla_qubit_idx, reset=True)

    def counting(dist, prefix):
        return len([x for x in dist if x[:len(prefix)] == prefix])

    def thresh_finding(dist):
        dist_list = list(map(lambda x: list(reversed(x[index_reg_qubits:])), dist))
        thresh = [0]
        last_floor = 0
        for b in range(length_dist ):
            curr_thresh_count = counting(dist_list,thresh) + last_floor
            if curr_thresh_count >= k:
                thresh.append(0)
            else:
                thresh[len(thresh)-1] = 1
                last_floor = curr_thresh_count
                if b < length_dist - 1:
                    thresh.append(0)
        return sum([2**(length_dist-i-1)*v for i, v in enumerate(thresh)])

    def bin_to_dec(bin, length_dist):
        return list(map(lambda x: sum([2**(length_dist-i-1)*v for i, v in enumerate(x)]), bin))


    def retrieve(dist,thresh):
        index, value = list(map(lambda x: x[:index_reg_qubits], dist)), list(map(lambda x: list(reversed(x[index_reg_qubits:])), dist))
        index_dec, value_dec = bin_to_dec(index, index_reg_qubits), bin_to_dec(value, length_dist)
        # print(value)
        # print("Values: ", value_dec)
        indices = [i for i, x in enumerate(value_dec) if x <= thresh]
        return [index_dec[i] for i in indices]


    device = qml.device("lightning.gpu", wires=number_of_qubits)

    @qml.qnode(device)
    def circuit():
        data_size = len(dataset)
        Prepare_index_register(index_reg, data_size)
        for p in range(P):
            query = query_list[p]
            data = list(map(lambda x: x[p], dataset))
            qml.BasisEmbedding(query, wires=query_reg)
            qml.QROM(data, index_reg, data_reg, ancilla_qubit, clean = True)
            Hamming_circuit(data_reg, query_reg, length)
            InC_circuit(data_reg, distance_reg, ancilla_qubit, length)
            qml.adjoint(Hamming_circuit)(data_reg, query_reg, length)
            qml.adjoint(qml.QROM)(data, index_reg, data_reg, ancilla_qubit, clean = True)
            qml.adjoint(qml.BasisEmbedding)(query, wires=query_reg)
        return qml.sample(wires = index_reg + distance_reg)

    # Get circuit depth and specs
    circuit_specs = qml.specs(circuit)()
    circuit_depth = circuit_specs['depth']
    circuit_num_gates = circuit_specs['num_operations']
    
    print(f"  🔬 Circuit specs - Shots: {shots}, Depth: {circuit_depth}, Gates: {circuit_num_gates}", flush=True)
    
    # Save circuit diagram if requested
    if save_circuit:
        try:
            circuit_dir = Path("circuits")
            circuit_dir.mkdir(exist_ok=True)
            
            fig, ax = qml.draw_mpl(circuit, decimals=2, style='sketch')(shots=shots)
            circuit_file = circuit_dir / f"circuit_query_{query_idx}.png"
            plt.savefig(circuit_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  💾 Circuit diagram saved to: {circuit_file}", flush=True)
        except Exception as e:
            print(f"  ⚠️ Could not save circuit diagram: {e}", flush=True)

    dist = [list(x) for x in dict.fromkeys(map(tuple, circuit(shots = shots)))]
    thresh = thresh_finding(dist)
    # print("Thresh: ", thresh)
    result = retrieve(dist,thresh)
    
    return result, shots, circuit_depth, circuit_num_gates

class QLSH:
    def __init__(self, input_dim, num_bits=16, num_tables=4, random_state=None):
        self.input_dim = input_dim
        self.num_bits = num_bits
        self.num_tables = num_tables
        self.rng = np.random.RandomState(random_state)

        self.hyperplanes = []
        for _ in range(num_tables):
            random_planes = self.rng.randn(num_bits, input_dim)
            Q, _ = np.linalg.qr(random_planes.T)
            self.hyperplanes.append(Q.T)  # Transpose back to original shape
        self.tables = [{} for _ in range(num_tables)]
        self.data = []  # List to store (vector, bit_array) tuples
        self.length = 4  # Length of each partition


    def _random_projection_data(self, data, hyperplane):
        """
        Process entire database at once
        data: array of shape (n_samples, input_dim)
        hyperplane: array of shape (num_bits, input_dim)
        Returns: list of lists, where each inner list contains binary strings for one sample
        """
        partition_length = 4
        # Project all data points at once: (num_bits, input_dim) @ (input_dim, n_samples) = (num_bits, n_samples)
        projections = np.dot(hyperplane, data.T)  # Shape: (num_bits, n_samples)
        binary_bits = (projections > 0).astype(int)  # Convert to 0s and 1s

        # Process each sample
        all_binary_strings = []
        for sample_idx in range(data.shape[0]):
            sample_bits = binary_bits[:, sample_idx]  # Get bits for this sample

            # Partition the binary bits into chunks of partition_length
            binary_strings = []
            for i in range(0, len(sample_bits), partition_length):
                chunk = sample_bits[i:i+partition_length]
                # Convert chunk to binary string
                binary_string = ''.join(map(str, chunk))
                binary_strings.append(binary_string)

            all_binary_strings.append(binary_strings)

        return all_binary_strings

    def _random_projection_query(self, x, hyperplane):
        """
        Process single query vector
        x: single vector of shape (input_dim,)
        hyperplane: array of shape (num_bits, input_dim)
        Returns: list of lists of integers for the single query vector
        """
        partition_length = 4
        # Project single query vector: (num_bits, input_dim) @ (input_dim,) = (num_bits,)
        projections = np.dot(hyperplane, x)
        binary_bits = (projections > 0).astype(int)  # Convert to 0s and 1s

        # Partition the binary bits into chunks of partition_length
        binary_partitions = []
        for i in range(0, len(binary_bits), partition_length):
            chunk = binary_bits[i:i+partition_length].tolist()  # Convert to list
            binary_partitions.append(chunk)

        return binary_partitions

    def build(self, data, bit_per_table=None):
        self.bit_per_table = bit_per_table if bit_per_table is not None else self.num_bits//2

        # Convert data to numpy array if it isn't already
        data_array = np.array(data)

        # Process all data at once for each hyperplane
        for table_idx, hyperplane in enumerate(self.hyperplanes):
            all_bit_arrays = self._random_projection_data(data_array, hyperplane)

            # Store data and update hash tables
            for i, (vector, bit_array) in enumerate(zip(data, all_bit_arrays)):
                if table_idx == 0:  # Only store the vector once
                    self.data.append((vector, [None] * self.num_tables))

                # Store bit array for this table
                self.data[i][1][table_idx] = bit_array

                # Update hash table using first bit_per_table bits (not partitions)
                # Concatenate all partitions to get the full bit string
                full_bit_string = ''.join(bit_array)
                # Take only the first bit_per_table bits
                hash_key = full_bit_string[:self.bit_per_table]
                self.tables[table_idx].setdefault(hash_key, []).append(i)
    def query(self, x, k, query_idx=0, save_circuit=True):
        """
        Query for k nearest neighbors using quantum Hamming distance
        x: query vector
        k: number of nearest neighbors to return
        query_idx: index of current query (for saving circuit)
        save_circuit: whether to save circuit diagram
        """

        candidates = set()

        # Get bit representations for query vector from all tables
        query_bits_all_tables = [self._random_projection_query(x, hyperplane) for hyperplane in self.hyperplanes]

        # Collect candidates from all hash tables
        for table_idx, (query_bits, table) in enumerate(zip(query_bits_all_tables, self.tables)):
            # Create hash key from first bit_per_table bits (not partitions)
            # Convert partitions to full bit string
            full_query_bits = []
            for partition in query_bits:
                full_query_bits.extend([str(bit) for bit in partition])
            full_query_string = ''.join(full_query_bits)
            # Take only the first bit_per_table bits
            hash_key = full_query_string[:self.bit_per_table]
            # Get candidates from this table
            table_candidates = table.get(hash_key, [])
            candidates.update(table_candidates)

        print(f"Number of candidates: {len(candidates)}", flush=True)

        if not candidates:
            # If no candidates found, return k random vectors
            random_indices = self.rng.choice(len(self.data), min(k, len(self.data)), replace=False)
            return [(self.data[i][0], float('inf')) for i in random_indices], 0, 0, 0

        # Prepare dataset for quantum computation
        # Convert candidate data to the format expected by Quantum_Hamming_Distance
        candidate_data = []

        for idx in candidates:
            vector, bit_arrays = self.data[idx]
            # bit_arrays is a list of lists of binary strings (one per table)
            # Flatten all binary strings from all tables into one list
            flattened_partitions = []
            for table_bits in bit_arrays:
                flattened_partitions.extend(table_bits)
            candidate_data.append(flattened_partitions)

        # Convert query bits to the same format
        # Flatten all partitions from all tables
        query_list = []
        for table_bits in query_bits_all_tables:
            for partition in table_bits:
                query_list.append(partition)  # Each partition is already a list of integers

        # Parameters for quantum computation
        length = self.length  # Length of each partition (4)

        # FIX: Calculate actual number of partitions
        actual_P = len(query_list)  # This is the actual number of partitions we have

        # Verify that candidate_data has the same structure
        if candidate_data:
            assert len(candidate_data[0]) == actual_P, f"Mismatch: candidate has {len(candidate_data[0])} partitions but query has {actual_P}"

        length_dist = int(np.ceil(np.log2(actual_P * length)))  # Length for distance register based on actual partitions

        print(f"Debug info:", flush=True)
        print(f"  num_tables: {self.num_tables}", flush=True)
        print(f"  num_bits: {self.num_bits}", flush=True)
        print(f"  partitions per table: {self.num_bits // self.length}", flush=True)
        print(f"  actual_P (total partitions): {actual_P}", flush=True)
        print(f"  query_list length: {len(query_list)}", flush=True)
        print(f"  candidate_data[0] length: {len(candidate_data[0]) if candidate_data else 'N/A'}", flush=True)

        # Call quantum Hamming distance function with circuit info
        quantum_results, shots, circuit_depth, circuit_num_gates = Quantum_Hamming_Distance(
            dataset=candidate_data,
            query_list=query_list,
            length=length,
            length_dist=length_dist,
            P=actual_P,  # Use actual number of partitions
            k=k,
            query_idx=query_idx,
            save_circuit=save_circuit
        )

        # Handle case where quantum function returns None
        if quantum_results is None:
            quantum_results = []

        # Convert quantum results back to (vector, distance) format
        results = []
        candidate_indices = list(candidates)
        used_original_indices = set()

        for quantum_idx in quantum_results:
            if quantum_idx < len(candidate_indices):
                original_idx = candidate_indices[quantum_idx]
                vector = self.data[original_idx][0]
                # Use quantum_idx as distance (lower means closer)
                results.append((vector, quantum_idx))
                used_original_indices.add(original_idx)

        print(f"Quantum returned {len(results)} results", flush=True)

        # If quantum returns fewer than k results, fill with additional candidates
        if len(results) < k:
            # Get remaining candidates not yet included
            remaining_candidates = [idx for idx in candidate_indices if idx not in used_original_indices]

            # Calculate cosine similarities for remaining candidates
            if remaining_candidates:
                remaining_vectors = [self.data[idx][0] for idx in remaining_candidates]
                similarities = cosine_similarity([x], remaining_vectors)[0]

                # Sort by similarity and take what we need
                sorted_indices = np.argsort(similarities)[::-1]
                needed = k - len(results)

                for i in sorted_indices[:needed]:
                    idx = remaining_candidates[i]
                    results.append((self.data[idx][0], -similarities[i]))

        # If quantum returns more than k results, sort by cosine similarity and take top k
        if len(results) > k:
            # Calculate cosine similarities for all quantum results
            vectors_for_similarity = [result[0] for result in results]
            similarities = cosine_similarity([x], vectors_for_similarity)[0]

            # Create list of (vector, quantum_distance, cosine_similarity) tuples
            enhanced_results = []
            for i, (vector, quantum_dist) in enumerate(results):
                enhanced_results.append((vector, quantum_dist, similarities[i]))

            # Sort by cosine similarity (descending - higher similarity is better)
            enhanced_results.sort(key=lambda x: x[2], reverse=True)

            # Take top k and convert back to (vector, distance) format
            results = [(vector, -cosine_sim) for vector, quantum_dist, cosine_sim in enhanced_results[:k]]

        return results[:k], shots, circuit_depth, circuit_num_gates



def get_cosine_ground_truth(data, query, k):
    """Calculate cosine similarity ground truth"""
    # Calculate cosine similarities between query and all data points
    similarities = cosine_similarity([query], data)[0]
    # Get indices of k most similar vectors (highest cosine similarity)
    nearest_neighbors = np.argsort(similarities)[-k:][::-1]  # Reverse for descending order
    return nearest_neighbors, similarities[nearest_neighbors]

def calculate_f1_score(qlsh_results, ground_truth_indices, qlsh_data):
    """Calculate F1 score between QLSH results and ground truth"""
    # Extract indices from QLSH results by finding their positions in the original data
    qlsh_indices = []
    for result_vector, _ in qlsh_results:
        # Find the index of this vector in the original dataset
        for i, (stored_vector, _) in enumerate(qlsh_data):
            if np.allclose(result_vector, stored_vector, atol=1e-10):
                qlsh_indices.append(i)
                break

    qlsh_set = set(qlsh_indices)
    ground_truth_set = set(ground_truth_indices)

    # Calculate precision, recall, and F1
    true_positives = len(qlsh_set & ground_truth_set)
    false_positives = len(qlsh_set - ground_truth_set)
    false_negatives = len(ground_truth_set - qlsh_set)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1, qlsh_indices

def format_time(seconds):
    """Format seconds to readable time string"""
    if seconds < 1:
        return f"{seconds*1000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        return str(timedelta(seconds=int(seconds)))

import time, json, datetime
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    load_digits,
    fetch_olivetti_faces,
    fetch_california_housing,
    fetch_20newsgroups_vectorized,
)
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import issparse

def _load_dataset_by_key(dataset_type: str):
    """
    Returns (X, y, dataset_name)
    """
    if dataset_type == 'iris':
        ds = load_iris()
        return ds.data, ds.target, "Iris"
    if dataset_type == 'wine':
        ds = load_wine()
        return ds.data, ds.target, "Wine"
    if dataset_type == 'breast_cancer':
        ds = load_breast_cancer()
        return ds.data, ds.target, "Breast Cancer"
    if dataset_type == 'digits':
        ds = load_digits()
        return ds.data, ds.target, "Digits"
    if dataset_type == 'faces':
        ds = fetch_olivetti_faces()
        X = ds.data
        y = getattr(ds, "target", None)
        if y is None:
            y = np.zeros(X.shape[0], dtype=int)
        return X, y, "Olivetti Faces"
    if dataset_type == 'housing':
        ds = fetch_california_housing()
        return ds.data, ds.target, "California Housing"
    if dataset_type == 'newsgroups_vec':
        Xy = fetch_20newsgroups_vectorized(subset='all', return_X_y=True)
        X, y = Xy
        if not issparse(X):
            X = np.asarray(X, dtype=np.float32)
        svd = TruncatedSVD(n_components=256, random_state=42)
        X_reduced = svd.fit_transform(X)
        y = np.asarray(y) if y is not None else np.zeros(X_reduced.shape[0], dtype=int)
        return X_reduced, y, "20 Newsgroups (vectorized, SVD=256)"
    raise ValueError(f"Unknown dataset type: {dataset_type}")

def test_qlsh_dataset(name=None, dataset_type='iris'):
    dataset_folder = Path("dataset")
    dataset_folder.mkdir(exist_ok=True)

    # Load dataset
    if name is not None:
        filepath = dataset_folder / name
        if not filepath.exists():
            print(f"Error: File {filepath} not found!", flush=True)
            return False

        print(f"Loading dataset from: {filepath}", flush=True)
        try:
            if filepath.suffix == '.csv':
                df = pd.read_csv(filepath)
            elif filepath.suffix in ['.xlsx', '.xls']:
                df = pd.read_excel(filepath)
            elif filepath.suffix == '.json':
                df = pd.read_json(filepath)
            elif filepath.suffix == '.parquet':
                df = pd.read_parquet(filepath)
            else:
                print(f"Unsupported file format: {filepath.suffix}", flush=True)
                return False

            print("Dataset loaded successfully!", flush=True)
            print(f"Shape: {df.shape}", flush=True)
            print(f"Columns: {df.columns.tolist()}", flush=True)
            if 'target' not in df.columns:
                df['target'] = 0
            dataset_name = f"CustomFile({name})"
            X = df.iloc[:, :-1].values
            y = df['target'].values
        except Exception as e:
            print(f"Error reading file: {e}", flush=True)
            return False
    else:
        try:
            X, y, dataset_name = _load_dataset_by_key(dataset_type)
        except Exception as e:
            print(f"Failed to load dataset '{dataset_type}': {e}", flush=True)
            return {'error': str(e)}

        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        df['target'] = y

        print(f"Testing QLSH with F1 Score Evaluation and Timing - {dataset_name} Dataset", flush=True)

    # Parameters
    n_samples = df.shape[0]
    n_dimensions = df.shape[1] - 1
    k = 5
    n_queries = min(3, n_samples)
    num_tables = 4
    num_bits = 8

    data = df.iloc[:, :-1].values.astype(float)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    data = data / norms

    rng = np.random.default_rng(123)
    query_indices = rng.choice(n_samples, n_queries, replace=False)
    queries = data[query_indices]

    print(f"\nDataset: {dataset_name}", flush=True)
    print(f"Shape: {data.shape} (samples, features)", flush=True)
    print(f"Queries: {n_queries} | Indices: {query_indices.tolist()}", flush=True)
    print(f"K (nearest neighbors): {k}", flush=True)

    # Initialize QLSH
    qlsh = QLSH(input_dim=n_dimensions, num_bits=num_bits, num_tables=num_tables, random_state=42)
    print(f"\nQLSH Configuration:", flush=True)
    print(f"  Input dimensions: {qlsh.input_dim}", flush=True)
    print(f"  Number of bits: {qlsh.num_bits}", flush=True)
    print(f"  Number of tables: {qlsh.num_tables}", flush=True)
    print(f"  Partition length: {qlsh.length}", flush=True)

    # Qubit count
    partitions_per_table = qlsh.num_bits // qlsh.length
    total_partitions = partitions_per_table * qlsh.num_tables
    length_dist = int(np.ceil(np.log2(total_partitions * qlsh.length)))
    index_reg_qubits = int(np.ceil(np.log2(len(data))))

    data_reg_qubits = qlsh.length
    query_reg_qubits = qlsh.length
    distance_reg_qubits = length_dist
    ancilla_qubits = 1
    total_qubits = index_reg_qubits + data_reg_qubits + query_reg_qubits + distance_reg_qubits + ancilla_qubits

    print(f"\n🔬 Quantum Circuit Configuration:", flush=True)
    print(f"  Index register qubits: {index_reg_qubits}", flush=True)
    print(f"  Data register qubits: {data_reg_qubits}", flush=True)
    print(f"  Query register qubits: {query_reg_qubits}", flush=True)
    print(f"  Distance register qubits: {distance_reg_qubits}", flush=True)
    print(f"  Ancilla qubits: {ancilla_qubits}", flush=True)
    print(f"  ➤ TOTAL QUBITS REQUIRED: {total_qubits}", flush=True)

    # Build index
    print(f"\nBuilding QLSH index...", flush=True)
    build_start = time.time()
    qlsh.build(data, bit_per_table=2)
    build_time = time.time() - build_start
    print(f"Build completed in {format_time(build_time)}", flush=True)
    print(f"  Data stored: {len(qlsh.data)} samples", flush=True)

    # Query + metrics
    print(f"\n{'='*70}\nTESTING {n_queries} QUERIES\n{'='*70}", flush=True)
    f1_scores, all_precisions, all_recalls, query_times = [], [], [], []
    all_shots, all_depths, all_gates = [], [], []
    epoch_start = time.time()

    for qi, query in enumerate(queries, 1):
        print(f"\n[Query {qi}/{n_queries}]", flush=True)
        
        # Ground truth
        gt_start = time.time()
        ground_truth_indices, _ = get_cosine_ground_truth(data, query, k)
        gt_time = time.time() - gt_start
        print(f"  Ground truth calculation: {format_time(gt_time)}", flush=True)

        # QLSH
        q_start = time.time()
        try:
            results, shots, circuit_depth, circuit_num_gates = qlsh.query(
                query, k, query_idx=qi, save_circuit=True
            )
            q_time = time.time() - q_start
            query_times.append(q_time)
            all_shots.append(shots)
            all_depths.append(circuit_depth)
            all_gates.append(circuit_num_gates)
            
            print(f"  QLSH query time: {format_time(q_time)}", flush=True)

            precision, recall, f1, _ = calculate_f1_score(results, ground_truth_indices, qlsh.data)
            f1_scores.append(f1)
            all_precisions.append(precision)
            all_recalls.append(recall)
            print(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}", flush=True)
        except Exception as e:
            q_time = time.time() - q_start
            query_times.append(q_time)
            all_shots.append(0)
            all_depths.append(0)
            all_gates.append(0)
            print(f"  Failed after {format_time(q_time)}: {str(e)[:120]}", flush=True)
            f1_scores.append(0.0)
            all_precisions.append(0.0)
            all_recalls.append(0.0)

    epoch_time = time.time() - epoch_start

    # Summary
    print(f"\n{'='*70}\nOVERALL RESULTS\n{'='*70}", flush=True)
    print(f"\n📊 Performance Metrics:", flush=True)
    print(f"  Average Precision: {np.mean(all_precisions):.4f} ± {np.std(all_precisions):.4f}", flush=True)
    print(f"  Average Recall   : {np.mean(all_recalls):.4f} ± {np.std(all_recalls):.4f}", flush=True)
    print(f"  Average F1 Score : {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}", flush=True)

    print(f"\n⏱️  Timing Statistics:", flush=True)
    print(f"  Build time       : {format_time(build_time)}", flush=True)
    print(f"  Total epoch time : {format_time(epoch_time)}", flush=True)
    print(f"  Average query    : {format_time(np.mean(query_times))} ± {format_time(np.std(query_times))}", flush=True)
    print(f"  Min / Max query  : {format_time(np.min(query_times))} / {format_time(np.max(query_times))}", flush=True)

    print(f"\n🔬 Circuit Statistics:", flush=True)
    print(f"  Average shots    : {np.mean(all_shots):.1f} ± {np.std(all_shots):.1f}", flush=True)
    print(f"  Average depth    : {np.mean(all_depths):.1f} ± {np.std(all_depths):.1f}", flush=True)
    print(f"  Average gates    : {np.mean(all_gates):.1f} ± {np.std(all_gates):.1f}", flush=True)
    print(f"  Min / Max shots  : {np.min(all_shots) if all_shots else 0} / {np.max(all_shots) if all_shots else 0}", flush=True)
    print(f"  Min / Max depth  : {np.min(all_depths) if all_depths else 0} / {np.max(all_depths) if all_depths else 0}", flush=True)

    print(f"\n💾 Memory Statistics:", flush=True)
    print(f"  Total samples indexed: {len(qlsh.data)}", flush=True)
    print(f"  Hash tables          : {len(qlsh.tables)}", flush=True)
    print(f"  Total qubits used    : {total_qubits}", flush=True)

    return {
        'build_time': build_time,
        'epoch_time': epoch_time,
        'query_times': query_times,
        'avg_query_time': float(np.mean(query_times)),
        'f1_scores': f1_scores,
        'avg_f1': float(np.mean(f1_scores)),
        'precisions': all_precisions,
        'recalls': all_recalls,
        'total_qubits': int(total_qubits),
        'qubit_breakdown': {
            'index_reg': int(index_reg_qubits),
            'data_reg': int(data_reg_qubits),
            'query_reg': int(query_reg_qubits),
            'distance_reg': int(distance_reg_qubits),
            'ancilla': int(ancilla_qubits)
        },
        'circuit_stats': {
            'avg_shots': float(np.mean(all_shots)) if all_shots else 0.0,
            'avg_depth': float(np.mean(all_depths)) if all_depths else 0.0,
            'avg_gates': float(np.mean(all_gates)) if all_gates else 0.0,
            'min_shots': int(np.min(all_shots)) if all_shots else 0,
            'max_shots': int(np.max(all_shots)) if all_shots else 0,
            'min_depth': int(np.min(all_depths)) if all_depths else 0,
            'max_depth': int(np.max(all_depths)) if all_depths else 0,
            'all_shots': [int(s) for s in all_shots],
            'all_depths': [int(d) for d in all_depths],
            'all_gates': [int(g) for g in all_gates]
        }
    }

def run_all_and_save():
    """
    Run all datasets and SORT FROM LARGE TO SMALL (descending order by sample size)
    Including: housing (~20.6k), newsgroups_vec (~18.8k), digits, breast_cancer
    """
    # Size hints for sorting (descending)
    size_hint = {
        "housing": 20640,
        "newsgroups_vec": 18846,
        "digits": 1797,
        "breast_cancer": 569,
    }

    wanted = ["housing", "newsgroups_vec", "digits", "breast_cancer"]
    # Sort descending (largest first)
    datasets_sorted = sorted(wanted, key=lambda k: size_hint[k], reverse=True)

    out_dir = Path("runs")
    out_dir.mkdir(exist_ok=True)
    out_txt = out_dir / "qlsh_all_datasets.txt"
    out_json = out_dir / "qlsh_all_datasets.json"

    start_time = datetime.datetime.now()
    print("=" * 100)
    print("=== 🧪 QLSH — Benchmark sklearn datasets (sorted LARGE → SMALL) ===")
    print("=" * 100)
    print(f"Order: {' → '.join(datasets_sorted)}")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)
    sys.stdout.flush()

    all_results = {}

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("=== QLSH — Benchmark sklearn datasets (sorted LARGE → SMALL) ===\n")
        f.write("=" * 100 + "\n")
        f.write(f"Order: {' → '.join(datasets_sorted)}\n")
        f.write(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.flush()

        for idx, ds in enumerate(datasets_sorted, 1):
            header = f"\n{'='*100}\n[{idx}/{len(datasets_sorted)}] DATASET: {ds.upper()}\n{'='*100}\n"
            print(header)
            f.write(header)
            f.flush()
            sys.stdout.flush()

            try:
                res = test_qlsh_dataset(dataset_type=ds)

                if isinstance(res, dict) and "error" in res:
                    all_results[ds] = {"error": res["error"]}
                    err_line = f"[ERROR] Skipped {ds}: {res['error']}\n"
                    print(err_line)
                    f.write(err_line)
                    f.flush()
                    sys.stdout.flush()
                    continue

                summary = {
                    "avg_f1": float(res.get("avg_f1", 0.0)),
                    "avg_query_time": float(res.get("avg_query_time", 0.0)),
                    "build_time": float(res.get("build_time", 0.0)),
                    "epoch_time": float(res.get("epoch_time", 0.0)),
                    "total_qubits": int(res.get("total_qubits", 0)),
                    "circuit_stats": res.get("circuit_stats", {})
                }
                all_results[ds] = summary

                circuit_stats = summary.get('circuit_stats', {})
                summary_text = (
                    f"\n--- SUMMARY for {ds.upper()} ---\n"
                    f"Average F1 score     : {summary['avg_f1']:.4f}\n"
                    f"Average query time   : {summary['avg_query_time']:.4f} s\n"
                    f"Build time           : {summary['build_time']:.4f} s\n"
                    f"Epoch time           : {summary['epoch_time']:.4f} s\n"
                    f"Total qubits used    : {summary['total_qubits']}\n"
                    f"\nCircuit Statistics:\n"
                    f"  Average shots      : {circuit_stats.get('avg_shots', 0):.1f}\n"
                    f"  Average depth      : {circuit_stats.get('avg_depth', 0):.1f}\n"
                    f"  Average gates      : {circuit_stats.get('avg_gates', 0):.1f}\n"
                    f"  Min/Max shots      : {circuit_stats.get('min_shots', 0)} / {circuit_stats.get('max_shots', 0)}\n"
                    f"  Min/Max depth      : {circuit_stats.get('min_depth', 0)} / {circuit_stats.get('max_depth', 0)}\n"
                )
                print(summary_text)
                f.write(summary_text)
                f.write("-" * 100 + "\n")
                f.flush()
                sys.stdout.flush()

            except Exception as e:
                err = f"[ERROR] {ds}: {e}\n"
                print(err)
                f.write(err)
                f.flush()
                sys.stdout.flush()
                all_results[ds] = {"error": str(e)}

    with open(out_json, "w", encoding="utf-8") as jf:
        json.dump(all_results, jf, ensure_ascii=False, indent=2)

    end_time = datetime.datetime.now()
    total_time = end_time - start_time
    footer = (
        f"\n{'='*100}\n"
        f"✅ ALL DATASETS COMPLETED\n"
        f"Start : {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"End   : {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Total time: {total_time}\n"
        f"📄 Detailed log saved to: {out_txt}\n"
        f"📊 Summary JSON saved to: {out_json}\n"
        f"🖼️  Circuit diagrams saved to: circuits/\n"
        f"{'='*100}\n"
    )
    print(footer)
    with open(out_txt, "a", encoding="utf-8") as f:
        f.write(footer)
    sys.stdout.flush()

if __name__ == "__main__":
    run_all_and_save()