# GCN Partitioning
Graph Partitoning Using Graph Convolutional Networks as described in [GAP: Generalizable Approximate Graph Partitioning Framework](https://arxiv.org/abs/1903.00614) 

Working based of off https://github.com/saurabhdash/GCN_Partitioning

## Loss Backward Equations
To handle large graphs, the Normalized Cuts loss function is implemented using sparse torch tensors using a custom loss class.

$[Z = (Y / \Gamma)(1 - Y)^{T} \circ A ]$

where $Y_{ij}$ is the probability of node i being in partition j.

$[L = \sum_{A_{lm} \neq 0} Z_{lm} ]$

Then the gradients can be calculated by the equations: 

$[\frac{\partial z_{i \alpha}}{\partial y_{ij}} = A_{i \alpha} \left(\frac{\Gamma_{j} (1 - y_{\alpha j}) - y_{ij}(1 - y_{\alpha j})D_{i}}{\Gamma_{j}^{2}}\right)]$

$[\frac{\partial z_{\alpha i}}{\partial y_{ij}} = A_{\alpha i} \left(\frac{\Gamma_{j} (- y_{\alpha j}) - y_{\alpha j}(1 - y_{ij})D_{i}}{\Gamma_{j}^{2}}\right)]$

$[\frac{\partial z_{i^{'} \alpha}}{\partial y_{ij}} = A_{i^{'} \alpha} \left(\frac{(1 - y_{\alpha j}) y_{i^{'}j}D_{i}}{\Gamma_{j}^{2}}\right) \;\;\; i^{'}, \alpha \neq i]$


The Balance loss is just calculated with different sums and a square so I just rely on autograd to calculate the backloss.
The loss is:

$[L = \sum_{reduce-sum} (1^T Y - \frac{n}{g})^2$

where n is the number of vertices and g is the number of partitions.


## Installation
Create a virtual environment using venv

```bash
python3 -m venv env
```

Source the virtual environment

```bash
source env/bin/activate
```

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.

```bash
pip install -r requirements.txt
```

## Usage
Check the notebook for how it is being used.

## Limitations
Has only been tested on random graphs with less than 100 vertices.
The balance loss currently makes the model predict the vertices to be equally likely for all partitions.
 - I tried to implement some sort of leakyReLU but it didn't help.
K-partitioning currently not working because without the use of balance it will just cut the graph into 2 partitions, and leave all the other partitions empty.