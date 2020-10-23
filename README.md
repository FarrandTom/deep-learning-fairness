## Neither Private Nor Fair: Impact of Data Imbalance on Utility and Fairness in Differential Privacy (Privacy-Preserving ML in Practice Workshop at CCS'20)
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

### References
[1] M. Abadi, A. Chu, I. Goodfellow, H. B. McMahan, I. Mironov, K. Talwar, and L. Zhang. Deep learning with differential privacy. In CCS, 2016.

[2] H. B. McMahan and G. Andrew. A general approach to adding differential privacy to iterative training procedures. arXiv:1812.06210, 2018

[3] H. B. McMahan, D. Ramage, K. Talwar, and L. Zhang. Learning differentially private recurrent language models. In ICLR, 2018

=======
