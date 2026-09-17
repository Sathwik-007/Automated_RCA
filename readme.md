# High-Scale Cloud RCA: Fast-Histogram Decision Tree in Pure NumPy

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-Optimized-green.svg)
![Machine Learning](https://img.shields.io/badge/ML-Whitebox_Explainability-orange.svg)

<img width="2636" height="1242" alt="Screen Recording 2026-09-16 at 10 59 43 PM 2" src="https://github.com/user-attachments/assets/96580f5a-8638-4ec9-b76b-159d0239fef7" />


A custom, dependency-free implementation of a Quantile-Binned, Fast-Histogram Decision Tree engineered from scratch in pure NumPy. This project performs highly explainable Root Cause Analysis (RCA) on massive, highly imbalanced cloud telemetry datasets by replicating the core performance engine of frameworks like LightGBM.

# Problem Overview
This project was inspired by a chaos engineering simulation where a targeted fault injection (an intentional bug introduced to a specific pod) took 3 hours to trace through manual log parsing. This engine was built to replace that manual review. By continuously monitoring structured telemetry, the model detects degrading metric thresholds and isolates the exact origin of the defect in milliseconds.

# The Dataset Architecture:
The engine is trained on the Cloud Computing Preprocessed Dataset from Hugging Face (specifically the 20231207 fault-injection subset).

The dataset is formatted specifically for high-performance machine learning rather than human readability:
- .npy (NumPy Binaries): The core telemetry is stored in pre-compiled NumPy arrays (e.g., pod_level_data_cpu_usage.npy, pod_level_data_rate_transmitted_packets.npy). This format allows for instantaneous loading into memory, bypassing the massive I/O bottlenecks of reading massive raw text or JSON logs.

- .csv (Metadata & Labels): Tabular files that act as the map for the binaries, containing timestamps, pod identifiers, and the binary ground-truth labels (0 for Normal, 1 for Anomaly).

- Feature Matrix: The final reconstructed matrix contains 1,194 continuous features tracking CPU, memory utilization, network bandwidth, and packet rates across the entire cluster architecture.

The dataset can be found here: [Link to HuggingFace dataset](https://huggingface.co/datasets/Lemma-RCA-NEC/Cloud_Computing_Preprocessed)

## 🚀 The Engineering Challenge & Optimization

Standard decision trees fail at scale on cloud metrics due to continuous variable sorting. Applying standard boolean masking (`X <= threshold`) across this dataset's matrix of **172,686 rows and 1,194 columns** requires approximately 52.5 billion array evaluations per node.

This implementation eliminates that bottleneck using two C-level memory optimizations:
1. **Quantile Binning:** Maps continuous floating-point telemetry into 255 discrete integer bins, eliminating $O(N \log N)$ sorting operations entirely.
2. **Fast Histogram Subtraction:** Scans the dataset exactly once to build a frequency histogram, then uses cumulative sums to evaluate split thresholds. This reduces the inner-loop time complexity from $O(N \times \text{Bins})$ to an $O(1)$ mathematical lookup.

*Benchmarked locally, this architectural optimization reduced total split-search operations from ~52.5B down to ~206M.*

<div align="center">
  <img src="./Automated_RCA/assets/performance_benchmark.png" alt="Algorithmic Time Complexity Reduction" width="600"/>
</div>

## 📊 Model Evaluation & Metrics

Cloud anomalies are inherently rare, meaning standard accuracy is a flawed metric. The model utilizes a custom weighted Gini impurity calculation (tuning the `anomaly_weight` parameter to 50.0) to force the tree to isolate minority-class anomalies without severe overfitting.

**Dataset:** 172,686 records | 1,194 Node/Pod metrics | Highly Imbalanced
**Business Objective:** 100% Recall on system failures with minimal false alarms to prevent alert fatigue.

| Metric | Score | Business Impact |
| :--- | :--- | :--- |
| **Recall** | `1.0000` | Zero critical anomalies missed in testing (72,287 / 72,287). |
| **F1-Score** | `0.9525` | Strong harmonic balance despite massive class imbalance. |
| **Precision** | `0.9094` | High confidence; false alarms are mathematically bounded. |
| **Accuracy** | `0.9583` | Overall system state correctly classified. |

<div align="center">
  <img src="./Automated_RCA/assets/confusion_matrix.png" alt="Confusion Matrix" width="400"/>
  <img src="./Automated_RCA/assets/pr_tradeoff.png" alt="Precision Recall Tradeoff" width="400"/>
</div>

## 🔍 Native White-Box Explainability Engine

In enterprise infrastructure, black-box anomaly detection is unusable for Site Reliability Engineers (SREs) during a live outage. They need to know exactly *why* an alert fired. 

Because this implementation is a singular, highly optimized decision tree, it operates as a fully transparent "white-box" model, eliminating the need for computationally heavy post-hoc approximation tools like SHAP. 

The architecture includes a built-in traversal engine that reverse-maps the C-optimized integer bins back to physical telemetry floats. It recursively traces the exact logical path a data point took through the tree's memory, outputting precise, human-readable RCA rules instantly.

**Example Console Output:**
```text
Analyzing Data Point at Index 724:
--- Prediction Explanation Path ---
 -> Moving [RIGHT] - Feature name: Cluster Level | Metric: Latency, Value: 61.0000 > Threshold: 52.5000
     -> Final Classification: ANOMALY

```

## 🛠️ Usage & Architecture

The architecture is fully object-oriented and contained within custom classes. No external ML libraries (e.g., scikit-learn (for decision-tree modelling), xgboost, lightgbm) are used.

*P.S: In this project scrikit-learn is only used to utilise confusion_matrix and RocCurveDisplay from the metrics module for visualisation purposes alone but never to use its model frameworks to build decision trees.*

```python
import numpy as np
from binner import QuantileBinner
from tree import QuantileDecisionTree

# 1. Transform raw telemetry to discrete bins
binner = QuantileBinner(max_bins=255)
binner.fit(X_raw)
X_binned = binner.transform(X_raw)

# 2. Train the Fast-Histogram Tree
tree = QuantileDecisionTree(max_depth=6, anomaly_weight=50.0, min_samples_leaf=5)
tree.fit(X_binned, y)

# 3. Predict and Explain
predictions = tree.predict(X_binned)
tree.explain_prediction(X_raw[idx], X_binned[idx], binner, feature_names)
```

Here is a clean, professional section you can append to your `README.md` to guide users on running the complete setup locally.

```markdown
## 💻 How to Run Locally

Since GitHub cannot host the 2.3 GB compiled model memory, you must download the artifacts before starting the engine.

**1. Download Model Artifacts**
* Download `artifacts.pkl` from [Insert Link to Google Drive/S3 here].
* Place the file directly in the root directory of this repository.

**2. Install Dependencies**
* Ensure you are using Python 3.10+.
* Install the required lightweight packages:
  ```bash
  pip install fastapi uvicorn numpy websockets

```

**3. Start the Streaming Server**

* Execute the main script to initialize the FastAPI Lifespan and load the model into memory:
```bash
python main.py

```



**4. View the Live Dashboard**

* Open your web browser and navigate to `http://localhost:8000`.
* The frontend will automatically establish a WebSocket connection and begin rendering the live telemetry and RCA traces.
