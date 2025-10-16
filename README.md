# The-excess-risk-bound-for-Multi-Graph-Dependent-MGD-linear-model-
This is a theoretical work. This project verifies the theoretical results by computing the excess risk bound for several multi-label datasets (such as emotions, yeast, corel5k, etc.).

# environment for experiments
recommended version: Python 3.9 or 3.10
install package, please execute the following command:
`pip install -r requirments`

add path for .py files: 
e.g. `configs/common_config.yaml`

note: our codes are implemented under Linux System.

# Datasets for experiments
please download the datasets to your device and edit the datasets path in `configs/common_config.yaml`.
url: 
(1).  http://mulan.sourceforge.net/datasets-mlc.html 
(2).  http://palm.seu.edu.cn/zhangml/

# running 
* train model and save, please run `pa_linear_model.py`.

* compute bound and r^*, please run `pa_compute_bound.py`.
