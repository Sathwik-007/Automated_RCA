import asyncio
import json
import math
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import pickle as pkl
import numpy as np
import sys

class NotebookUnpickler(pkl.Unpickler):
    def find_class(self, module, name):
        if module == '__main__':
            # Force it to look in the current file for the class
            return getattr(sys.modules['__main__'], name)
        return super().find_class(module, name)

class Node:
    def __init__(self, left=None, right=None, best_feature_to_split=None, threshold_bin=None, *, value=None):
        self.best_feature_to_split = best_feature_to_split
        self.threshold_bin = threshold_bin
        self.left = left
        self.right = right

        self.value = value

        
    def is_leaf_node(self):
        return self.value is not None

class QuantileBinner:
    '''Implementation of Quanitle Binning concept for improved performance

        Purpose: to create bins into which the data points might fall into and append them to the global list @self.bins_per_feature
    '''

    def __init__(self, max_bins=255):
        self.max_bins = max_bins
        self.bins_per_feature = []

    def fit(self, X):
        
        n_features = X.shape[1]

        for feat in range(n_features):
            col = X[:, feat]
            unique_values = np.unique(col)

            if len(unique_values) < self.max_bins:
                self.bins_per_feature.append(unique_values)

            else:
                percentiles = np.linspace(0, 100, self.max_bins)
                bins = np.percentile(col, percentiles)
                self.bins_per_feature.append(np.unique(bins))

        return self

    def transform(self, X):
        # create a matrix like X but dtype strictly to be uint32
        X_binned = np.zeros_like(X, dtype=np.uint32)
        n_features = X.shape[1]
        for feat in range(n_features):
            # X_binned is going to contain: where all the actual values of X at that feature lie in that corresponding bin at that index
            # dictated by self.bins_per_feature (a list of lists containing bins for a feature corresponding to index)
            X_binned[:, feat] = np.searchsorted(self.bins_per_feature[feat], X[:, feat])
            
        return X_binned

class QuantileDecisionTree:
    def __init__(self, max_depth=10, min_samples_leaf=5, min_gain=1e-4, anomaly_weight=10.0):
        self.max_depth = max_depth
        self.max_bins_per_feature = []
        self.min_samples_leaf = min_samples_leaf
        self._n_classes = 2
        self.root = None
        self.anomaly_weight = anomaly_weight
        self.min_gain = min_gain

    def _weighted_gini(self, y):
        m = len(y)
        if m == 0:
            return 0.0
        counts = np.bincount(y, minlength=self._n_classes)
        weighted_counts = np.array([counts[0], counts[1] * self.anomaly_weight])

        total_weight = np.sum(weighted_counts)

        if total_weight == 0.0:
            return 0.0

        probs = weighted_counts / total_weight
        return 1.0 - np.sum(probs ** 2)

    def _best_split(self, X_binned, y):
        n_samples, n_features = X_binned.shape
        best_gain = self.min_gain

        best_feature_idx, best_threshold = None, None

        parent_gini = self._weighted_gini(y)
        total_counts = np.bincount(y, minlength=self._n_classes)

        for feat in range(n_features):
            feat_data = X_binned[:, feat]

            max_bins = self.max_bins_per_feature[feat]

            if max_bins == 0:
                continue
                
            bincount_0 = np.bincount(feat_data[y == 0], minlength=max_bins+1)
            bincount_1 = np.bincount(feat_data[y == 1], minlength=max_bins+1)

            cumsum_0 = np.cumsum(bincount_0)
            cumsum_1 = np.cumsum(bincount_1)

            for thresh in range(max_bins):
                n_left_0 = cumsum_0[thresh]
                n_left_1 = cumsum_1[thresh]

                n_right_0 = total_counts[0] - n_left_0
                n_right_1 = total_counts[1] - n_left_1

                n_left = n_left_0 + n_left_1
                n_right = n_right_0 + n_right_1

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                wl = np.array([n_left_0, n_left_1 * self.anomaly_weight])
                wr = np.array([n_right_0, n_right_1 * self.anomaly_weight])

                sum_wl = np.sum(wl)
                sum_wr = np.sum(wr)

                gini_left = 1.0 - np.sum((wl / sum_wl) ** 2) if sum_wl > 0 else 0.0
                gini_right = 1.0 - np.sum((wr / sum_wr) ** 2) if sum_wr > 0 else 0.0

                child_gini = (n_left / n_samples) * gini_left + (n_right / n_samples) * gini_right

                information_gain = parent_gini - child_gini

                if information_gain > best_gain:
                    best_feature_idx = feat
                    best_threshold = thresh
                    best_gain = information_gain

        return best_feature_idx, best_threshold

        
    def _compute_leaf_value(self, y):
        return np.argmax(np.bincount(y, minlength=self._n_classes))
                
    
    def _build_tree(self, X_binned, y, depth=0):

        # recursion stop conditions
        if depth > self.max_depth or len(np.unique(y)) == 1:
            leaf_value = self._compute_leaf_value(y)
            return Node(value=leaf_value)

        best_feature_idx, best_threshold = self._best_split(X_binned, y)

        if best_feature_idx is None:
            leaf_value = self._compute_leaf_value(y)
            return Node(value=leaf_value)

        left_mask = X_binned[:, best_feature_idx] <= best_threshold
        right_mask = ~left_mask

        y_left, y_right = y[left_mask], y[right_mask]

        left_child = self._build_tree(X_binned[left_mask], y_left, depth+1)
        right_child = self._build_tree(X_binned[right_mask], y_right, depth+1)

        return Node(left=left_child, right=right_child, best_feature_to_split=best_feature_idx, threshold_bin=best_threshold)

    def fit(self, X_binned, y):
        n_features = X_binned.shape[1]
        self.max_bins_per_feature = [np.max(X_binned[:, feat]) if len(X_binned[:, feat]) > 0 else 0 for feat in range(n_features)]
        self.root = self._build_tree(X_binned, y, depth=0)

    def predict(self, X_binned):
        return np.array([self._predict_row(x, self.root) for x in X_binned])

    def _predict_row(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.best_feature_to_split] <= node.threshold_bin:
            return self._predict_row(x, node.left)

        else:
            return self._predict_row(x, node.right)

    def explain_prediction(self, x_raw, x_binned, binner, feature_names, node=None, depth=0, path=None):
        if node is None:
            node = self.root
            path = ["--- Prediction Explanation Path ---"]

        if node.value is not None:
            status = 'ANOMALY' if node.value == 1 else 'NORMAL'
            path.append(f'{"    " * depth} -> Final Classification: {status}')
            return "\n".join(path)

        feat_name = feature_names[node.best_feature_to_split]
        actual_value = x_raw[node.best_feature_to_split]

        split_value = binner.bins_per_feature[node.best_feature_to_split][node.threshold_bin]

        if x_binned[node.best_feature_to_split] <= node.threshold_bin:
            path.append(f'{"    " * depth} -> Moving [LEFT] - Feature name : {feat_name}, Value : {actual_value:.4f} <= Threshold : {split_value:.4f}')
            return self.explain_prediction(x_raw, x_binned, binner, feature_names, node.left, depth + 1, path)

        else:
            path.append(f'{"    " * depth} -> Moving [RIGHT] - Feature name : {feat_name}, Value : {actual_value:.4f} > Threshold : {split_value:.4f}')
            return self.explain_prediction(x_raw, x_binned, binner, feature_names, node.right, depth + 1, path)

ml_models = {}

async def lifespan(app: FastAPI):
    global ml_models
    with open("./artifacts.pkl", "rb") as f:
        model_artifacts = pkl.load(f)

    ml_models = {
        "X_raw": model_artifacts["X_raw"],
        "X_binned": model_artifacts["X_binned"],
        "tree": model_artifacts["tree"],
        "binner": model_artifacts["binner"],
        "feature_names": model_artifacts["feature_names"]
    }
    
    yield
    # Shutdown: Clean up memory
    print("Shutting down RCA Engine...")
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    start_idx = 38
    
    try:
        # 4. Reference ml_models dictionary instead of raw variables
        for idx in range(start_idx, len(ml_models["X_raw"])):
            row_binned = ml_models["X_binned"][idx].reshape(1, -1)
            row_raw = ml_models["X_raw"][idx]
            
            prediction = ml_models["tree"].predict(row_binned)[0]
            
            raw_val = float(np.max(row_raw))
            chart_metric = raw_val if not (math.isnan(raw_val) or math.isinf(raw_val)) else 0.0
            
            payload = {
                "timestamp": idx,
                "metric_value": chart_metric,
                "status": "NORMAL",
                "rca_trace": None
            }
            
            if prediction == 1:
                payload["status"] = "ANOMALY"
                payload["rca_trace"] = ml_models["tree"].explain_prediction(
                    x_raw=row_raw, 
                    x_binned=row_binned[0], 
                    binner=ml_models["binner"], 
                    feature_names=ml_models["feature_names"]
                )
                
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(1) 
            
    except WebSocketDisconnect:
        print("Client disconnected cleanly.")

@app.get("/")
async def get():
    with open("index.html", "r") as f:
        return HTMLResponse(f.read())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)