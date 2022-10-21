# Forecasting Interaction Order on Temporal Graphs (KDD21)
#### Authors: Wenwen Xia, Yuchen Li, Jianwei Tian, and Shenghong Li
#### Please contact xiawenwen@sjtu.edu.cn for any questions.

 
## Abstract
Link prediction is a fundamental task of graph analysis and the topic has been studied extensively in the case of static graphs. Recent research interests in predicting links/interactions on temporal graphs. 
Most prior works employ snapshots to model graph dynamics and formulate the binary classification problem to predict the existence of a future edge. 
However, the binary formulation ignores the order of interactions and thus fails to capture the fine-grained temporal information in the data. 
In this paper, we propose a new problem on temporal graphs to predict the interaction order for a given node set (IOD).
We develop a Temporal ATtention network (TAT) for the IOD problem.
TAT utilizes fine-grained time information by encoding continuous time as fixed-length feature vectors.
For each transformation layer of TAT, we adopt attention mechanism to compute adaptive aggregations based on former layer's node representations and encoded time vectors.
We also devise a novel training scheme for TAT to address the permutation-sensitive property of IOD.
Experiments on several real-world temporal networks reveal that TAT outperforms the state-of-the-art graph neural networks by 55\% on average under the AUC metric.


# Requirements

* python >= 3.7

* Dependency

```{bash}
scipy==1.5.0
numpy==1.18.5
scikit_learn==0.23.2
torch==1.9.0
torch_geometric==1.7.2
networkx==2.8.7
torch_sparse==0.6.15
torch_scatter==2.0.9
```

# Usage
1. Install
```{bash}
git clone https://github.com/xiawenwen49/TAT-code.git
cd TAT-code/
```
```python
pip install -e . # first install the code as a local editable module
```
2. Preprocessing a dataset
```python
python -m TAT.preprocessing --dataset CollegeMsg
```
3. Run the model
```python
python -m TAT.main --dataset CollegeMsg --model TAT --gpu 0
```


# Other parameters
```python
python -m TAT.main --help
```

