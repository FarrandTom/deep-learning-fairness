## Neither Private Nor Fair: Impact of Data Imbalance on Utility and Fairness in Differential Privacy (CCS'20 Privacy-Preserving ML in Practice Workshop)
The paper discusses how Differential Privacy (specifically DPSGD from [1]) impacts model performance for underrepresented groups. 
We aim to study how different levels of imbalance in the data affect the accuracy and the fairness of the decisions made by the model, given different levels of privacy. We demonstrate how even small imbalances and loose privacy guarantees can cause disparate impacts.

### Usage
Configure environment by running: `pip install -r requirements.txt` <br />
We use Python3.7 and GPU Nvidia TitanX. <br />
File `playing.py` serves as the entry point for the code. It uses `utils/params.yaml` to set parameters from the paper and builds a graph on Tensorboard. <br />
For Sentiment prediction we use `playing_nlp.py`.


### Datasets
1. MNIST (part of PyTorch)
2. Diversity in Faces (obtained from IBM [here](https://www.research.ibm.com/artificial-intelligence/trusted-ai/diversity-in-faces/#access))
3. iNaturalist (download from [here](https://github.com/visipedia/inat_comp))
4. UTKFace (from [here](http://aicip.eecs.utk.edu/wiki/UTKFace))
5. AAE Twitter corpus (from [here](http://slanglab.cs.umass.edu/TwitterAAE/))

### Code Sources
We use `compute_dp_sgd_privacy.py` copied from public [repo](https://github.com/tensorflow/privacy).

DP-FedAvg implementation is taken from public [repo](https://github.com/ebagdasa/backdoor_federated_learning).  

Implementation of DPSGD is based on TF Privacy [repo](https://github.com/tensorflow/privacy) and papers:

### Paper
https://arxiv.org/pdf/2009.06389.pdf

### Citation
`@article{farrand2020neither,
  title={Neither Private Nor Fair: Impact of Data Imbalance on Utility and Fairness in Differential Privacy},
  author={Farrand, Tom and Mireshghallah, Fatemehsadat and Singh, Sahib and Trask, Andrew},
  journal={arXiv preprint arXiv:2009.06389},
  year={2020}
}`

=======
