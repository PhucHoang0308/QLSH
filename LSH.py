import numpy as np
import pandas as pd
import timeit

class LSH:
    def __init__(self, input_dim, num_bits=16, num_tables=4, random_state=None):
        self.input_dim = input_dim
        self.num_bits = num_bits
        self.num_tables = num_tables
        self.rng = np.random.RandomState(random_state)
        
        # Create multiple tables with different random projections using list comprehensions
        # self.hyperplanes = [self.rng.randn(num_bits, input_dim) for _ in range(num_tables)]
        self.hyperplanes = []
        for _ in range(num_tables):
            random_planes = self.rng.randn(num_bits, input_dim)
            Q, _ = np.linalg.qr(random_planes.T)
            self.hyperplanes.append(Q.T)  # Transpose back to original shape
        self.tables = [{} for _ in range(num_tables)]
        self.data = []  # List to store (vector, bit_array) tuples

    def _random_projection(self, x, hyperplane):
        projections = np.dot(hyperplane, x)
        return [1 if p > 0 else 0 for p in projections]
    
    def _hamming_distance(self, b1, b2):
        return sum(el1 != el2 for el1, el2 in zip(b1, b2))

    def build(self, data, bit_per_table=None):
        self.bit_per_table = bit_per_table if bit_per_table is not None else self.num_bits//2
        for i, vector in enumerate(data):
            # Store vector and its bit representations
            bit_arrays = [self._random_projection(vector, hyperplane) for hyperplane in self.hyperplanes]
            self.data.append((vector, bit_arrays))
            
            # Update hash tables
            [table.setdefault(tuple(bit_array[:self.bit_per_table]), []).append(i) 
             for bit_array, table in zip(bit_arrays, self.tables)]

    def query(self, x, k):
        candidates = set()
        
        s = timeit.default_timer()
        # Get bit representations for query vector
        query_bits = [self._random_projection(x, hyperplane) for hyperplane in self.hyperplanes]
        e = timeit.default_timer()
        #print(f"Hashing time: {round((e - s) * 1e6, 3)} µs")

        # Collect candidates from all tables
        [candidates.update(table.get(tuple(bits[:self.bit_per_table]), [])) 
         for bits, table in zip(query_bits, self.tables)]
        #print("Total number of candidates:", len(candidates))

        if not candidates:
            # If no candidates found, return k random vectors
            random_indices = self.rng.choice(len(self.data), min(k, len(self.data)), replace=False)
            return [(self.data[i][0], float('inf')) for i in random_indices]
        
        # Calculate Hamming distances for candidates using all bit arrays
        results = [(self.data[idx][0], 
                   sum(self._hamming_distance(q, c) for q, c in zip(query_bits, self.data[idx][1]))) 
                  for idx in candidates]

        #print(f"Hamming time: {round((e1 - s1) * 1e6, 3)} µs")

        # Sort by Hamming distance and return top k
        return sorted(results, key=lambda pair: pair[1])[:k]
# write main code using data from sklearn which are Iris, Breast Cancer, Digits, fetch_covtype. Having the results K neighbors and F1-score
if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler

    # Load datasets
    datasets_list = {
        "Iris": datasets.load_iris(),
        "Breast Cancer": datasets.load_breast_cancer(),
        "Digits": datasets.load_digits(),
        "Covtype": datasets.fetch_covtype()
    }

    for name, dataset in datasets_list.items():
        print(f"\nDataset: {name}")
        X = dataset.data
        y = dataset.target

        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Initialize and build LSH
        lsh = LSH(input_dim=X.shape[1], num_bits=16, num_tables=4, random_state=42)
        lsh.build(X_train)

        # Query for nearest neighbors and predict labels
        k = 5
        y_pred = []
        for x in X_test:
            neighbors = lsh.query(x, k)
            neighbor_indices = [np.where((X_train == neighbor[0]).all(axis=1))[0][0] for neighbor in neighbors]
            neighbor_labels = [y_train[idx] for idx in neighbor_indices]
            # Majority vote
            pred_label = max(set(neighbor_labels), key=neighbor_labels.count)
            y_pred.append(pred_label)

        # Calculate F1-score
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"F1-score: {f1:.4f}")
