
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

def Quantum_Hamming_Distance(dataset, query_list, length, length_dist, P, k):

    """
    dataset: List of size N of the P partitions lists of samples
    query_list: List of size P of query paritions
    length: Length of 1 partition
    P: Number of partition per sample
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

    dist = [list(x) for x in dict.fromkeys(map(tuple, circuit(shots = shots)))]
    thresh = thresh_finding(dist)
    # print("Thresh: ", thresh)
    result = retrieve(dist,thresh)
    return result

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
    def query(self, x, k):
        """
        Query for k nearest neighbors using quantum Hamming distance
        x: query vector
        k: number of nearest neighbors to return
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

        print(f"Number of candidates: {len(candidates)}")

        if not candidates:
            # If no candidates found, return k random vectors
            random_indices = self.rng.choice(len(self.data), min(k, len(self.data)), replace=False)
            return [(self.data[i][0], float('inf')) for i in random_indices]

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

        print(f"Debug info:")
        print(f"  num_tables: {self.num_tables}")
        print(f"  num_bits: {self.num_bits}")
        print(f"  partitions per table: {self.num_bits // self.length}")
        print(f"  actual_P (total partitions): {actual_P}")
        print(f"  query_list length: {len(query_list)}")
        print(f"  candidate_data[0] length: {len(candidate_data[0]) if candidate_data else 'N/A'}")

        # Call quantum Hamming distance function
        quantum_results = Quantum_Hamming_Distance(
            dataset=candidate_data,
            query_list=query_list,
            length=length,
            length_dist=length_dist,
            P=actual_P,  # Use actual number of partitions
            k=k
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

        print(f"Quantum returned {len(results)} results")

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

        return results[:k]



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

def test_qlsh_dataset(name=None, dataset_type='iris'):
    dataset_folder = Path("dataset")
    dataset_folder.mkdir(exist_ok=True)

    if name is not None:
        filepath = dataset_folder / name

        if not filepath.exists():
            print(f"Error: File {filepath} not found!")
            return False

        print(f"Loading dataset from: {filepath}")

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
                print(f"Unsupported file format: {filepath.suffix}")
                return False

            print(f"Dataset loaded successfully!")
            print(f"Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")

        except Exception as e:
            print(f"Error reading file: {e}")
            return False
    else:
        # Load dataset mặc định theo dataset_type
        if dataset_type == 'iris':
            from sklearn.datasets import load_iris
            dataset = load_iris()
            dataset_name = "Iris"
        elif dataset_type == 'breast_cancer':
            from sklearn.datasets import load_breast_cancer
            dataset = load_breast_cancer()
            dataset_name = "Breast Cancer"
        elif dataset_type == 'wine':
            from sklearn.datasets import load_wine
            dataset = load_wine()
            dataset_name = "Wine"
        elif dataset_type == 'digits':
            from sklearn.datasets import load_digits
            dataset = load_digits()
            dataset_name = "Digits"
        else:
            print(f"Unknown dataset type: {dataset_type}")
            print("Available types: iris, breast_cancer, wine, digits")
            return False

        df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
        df['target'] = dataset.target
        print(f"Testing QLSH with F1 Score Evaluation and Timing - {dataset_name} Dataset")

    # Dataset parameters
    n_samples = df.shape[0]
    n_dimensions = df.shape[1] - 1  # Exclude target column
    k = 5
    n_queries = 3
    num_tables = 4
    num_bits = 8

    # Prepare dataset
    data = df.iloc[:, :-1].values  # Features only
    # Normalize to unit vectors
    data = data / np.linalg.norm(data, axis=1)[:, np.newaxis]

    # Select random query vectors from the dataset
    np.random.seed(123)
    query_indices = np.random.choice(n_samples, n_queries, replace=False)
    queries = data[query_indices]

    print(f"\nDataset shape: {data.shape}")
    print(f"Number of queries: {n_queries}")
    print(f"Queries are selected from dataset indices: {query_indices}")
    print(f"K (nearest neighbors): {k}")

    # Initialize QLSH with dataset parameters
    qlsh = QLSH(input_dim=n_dimensions, num_bits=num_bits, num_tables=num_tables, random_state=42)

    print(f"\nQLSH Configuration:")
    print(f"  Input dimensions: {qlsh.input_dim}")
    print(f"  Number of bits: {qlsh.num_bits}")
    print(f"  Number of tables: {qlsh.num_tables}")
    print(f"  Partition length: {qlsh.length}")

    # Calculate qubit requirements
    partitions_per_table = qlsh.num_bits // qlsh.length
    total_partitions = partitions_per_table * qlsh.num_tables
    length_dist = int(np.ceil(np.log2(total_partitions * qlsh.length)))
    index_reg_qubits = int(np.ceil(np.log2(len(data))))

    data_reg_qubits = qlsh.length
    query_reg_qubits = qlsh.length
    distance_reg_qubits = length_dist
    ancilla_qubits = 1
    total_qubits = index_reg_qubits + data_reg_qubits + query_reg_qubits + distance_reg_qubits + ancilla_qubits

    print(f"\n🔬 Quantum Circuit Configuration:")
    print(f"  Index register qubits: {index_reg_qubits}")
    print(f"  Data register qubits: {data_reg_qubits}")
    print(f"  Query register qubits: {query_reg_qubits}")
    print(f"  Distance register qubits: {distance_reg_qubits}")
    print(f"  Ancilla qubits: {ancilla_qubits}")
    print(f"  ➤ TOTAL QUBITS REQUIRED: {total_qubits}")

    # Build the index with timing
    print(f"\nBuilding QLSH index...")
    build_start = time.time()
    qlsh.build(data, bit_per_table=2)
    build_time = time.time() - build_start

    print(f"Build completed in {format_time(build_time)}")
    print(f"  Data stored: {len(qlsh.data)} samples")

    # Test multiple queries and collect metrics
    print(f"\n{'='*70}")
    print(f"TESTING {n_queries} QUERIES")
    print(f"{'='*70}")

    f1_scores = []
    all_precisions = []
    all_recalls = []
    query_times = []

    # Start total epoch timing
    epoch_start = time.time()

    for query_idx, query in enumerate(queries):
        print(f"\n[Query {query_idx + 1}/{n_queries}]")

        # Time for ground truth calculation
        gt_start = time.time()
        ground_truth_indices, ground_truth_similarities = get_cosine_ground_truth(data, query, k)
        gt_time = time.time() - gt_start
        print(f"  Ground truth calculation: {format_time(gt_time)}")

        # Time for QLSH query
        query_start = time.time()
        try:
            results = qlsh.query(query, k)
            query_time = time.time() - query_start
            query_times.append(query_time)

            print(f"  QLSH query time: {format_time(query_time)}")

            # Calculate F1 score
            precision, recall, f1, qlsh_indices = calculate_f1_score(results, ground_truth_indices, qlsh.data)

            # Store metrics
            f1_scores.append(f1)
            all_precisions.append(precision)
            all_recalls.append(recall)

            print(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

        except Exception as e:
            query_time = time.time() - query_start
            query_times.append(query_time)
            print(f"  Failed after {format_time(query_time)}: {str(e)[:100]}")
            f1_scores.append(0.0)
            all_precisions.append(0.0)
            all_recalls.append(0.0)

    # End total epoch timing
    epoch_time = time.time() - epoch_start

    # Calculate and display results
    print(f"\n{'='*70}")
    print(f"OVERALL RESULTS")
    print(f"{'='*70}")

    print(f"\n📊 Performance Metrics:")
    print(f"  Average Precision: {np.mean(all_precisions):.4f} ± {np.std(all_precisions):.4f}")
    print(f"  Average Recall: {np.mean(all_recalls):.4f} ± {np.std(all_recalls):.4f}")
    print(f"  Average F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

    print(f"\n⏱️  Timing Statistics:")
    print(f"  Build time: {format_time(build_time)}")
    print(f"  Total epoch time: {format_time(epoch_time)}")
    print(f"  Average query time: {format_time(np.mean(query_times))} ± {format_time(np.std(query_times))}")
    print(f"  Min query time: {format_time(np.min(query_times))}")
    print(f"  Max query time: {format_time(np.max(query_times))}")

    print(f"\n📋 Individual Query Times:")
    for i, qt in enumerate(query_times):
        print(f"  Query {i+1}: {format_time(qt)} (F1: {f1_scores[i]:.4f})")

    print(f"\n💾 Memory Statistics:")
    print(f"  Total samples indexed: {len(qlsh.data)}")
    print(f"  Hash tables: {len(qlsh.tables)}")
    print(f"  Total qubits used: {total_qubits}")

    return {
        'build_time': build_time,
        'epoch_time': epoch_time,
        'query_times': query_times,
        'avg_query_time': np.mean(query_times),
        'f1_scores': f1_scores,
        'avg_f1': np.mean(f1_scores),
        'precisions': all_precisions,
        'recalls': all_recalls,
        'total_qubits': total_qubits,
        'qubit_breakdown': {
            'index_reg': index_reg_qubits,
            'data_reg': data_reg_qubits,
            'query_reg': query_reg_qubits,
            'distance_reg': distance_reg_qubits,
            'ancilla': ancilla_qubits
        }
    }
from contextlib import redirect_stdout
import io, json

def run_all_and_save():
    datasets = ["iris", "breast_cancer", "wine", "digits"]
    out_dir = Path("runs"); out_dir.mkdir(exist_ok=True)
    out_txt = out_dir / "qlsh_all_datasets.txt"     # toàn bộ log + tóm tắt
    out_json = out_dir / "qlsh_all_datasets.json"   # summary dạng JSON (tiện phân tích thêm)

    all_results = {}
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("=== QLSH – Benchmark all sklearn datasets ===\n")
        f.write("Datasets: " + ", ".join(datasets) + "\n\n")

        for ds in datasets:
            f.write(f"\n{'='*80}\n")
            f.write(f"DATASET: {ds}\n")
            f.write(f"{'='*80}\n")

            buf = io.StringIO()
            try:
                # redirect toàn bộ stdout của test_qlsh_dataset vào buffer
                with redirect_stdout(buf):
                    res = test_qlsh_dataset(dataset_type=ds)
                log = buf.getvalue()
                f.write(log)  # ghi toàn bộ log chi tiết

                # lưu summary gọn để tra cứu nhanh
                all_results[ds] = {
                    "avg_f1": float(res.get("avg_f1", 0.0)),
                    "avg_query_time": float(res.get("avg_query_time", 0.0)),
                    "build_time": float(res.get("build_time", 0.0)),
                    "epoch_time": float(res.get("epoch_time", 0.0)),
                    "total_qubits": int(res.get("total_qubits", 0)),
                }

                # ghi tóm tắt cuối mỗi dataset vào .txt
                f.write("\n--- SUMMARY ---\n")
                f.write(json.dumps(all_results[ds], ensure_ascii=False, indent=2))
                f.write("\n")

            except Exception as e:
                log = buf.getvalue()
                f.write(log)
                f.write(f"\n[ERROR] {ds}: {e}\n")
                all_results[ds] = {"error": str(e)}

    # ghi thêm 1 file JSON tổng hợp
    with open(out_json, "w", encoding="utf-8") as jf:
        json.dump(all_results, jf, ensure_ascii=False, indent=2)

    print(f"\n✅ Done. Saved logs to: {out_txt}")
    print(f"✅ JSON summary: {out_json}")

if __name__ == "__main__":
    # nếu trước đó có dòng demo đơn lẻ thì comment lại:
    # results = test_qlsh_dataset(dataset_type='digits'); print(results)

    # chạy toàn bộ
    run_all_and_save()