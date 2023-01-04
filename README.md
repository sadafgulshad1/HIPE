# Hierarchical-ProtoPNet
This repository contains the official implementation of our work [Hierarchical Explanations for Video Action Recognition](https://arxiv.org/pdf/2301.00436.pdf).

**Abstract:** *We propose Hierarchical ProtoPNet: an interpretable network that explains its reasoning process by considering the hierarchical relationship between classes. Different from previous methods that explain their reasoning process by dissecting the input image and finding the prototypical parts responsible for the classification, we propose to explain the reasoning process for video action classification by dissecting the input video frames on multiple levels of the class hierarchy. The explanations leverage the hierarchy to deal with uncertainty, akin to human reasoning: When we observe water and human activity, but no definitive action it can be recognized as the water sports parent class. Only after observing a person swimming can we definitively refine it to the swimming action. Experiments on ActivityNet and UCF-101 show performance improvements while providing multi-level explanations.* 

<img src="https://github.com/sadafgulshad1/Hierarchical-ProtoPNet/blob/main/Architecture_HProtoPNet.png"  />

## Dataset Preparation
### UCF-101
*
*
## BibTeX
If you found this work useful in your research, please consider citing
```
@article{gulshad2023hierarchical,
  title={Hierarchical Explanations for Video Action Recognition},
  author={Gulshad, Sadaf and Long, Teng and van Noord, Nanne},
  journal={arXiv preprint arXiv:2301.00436},
  year={2023}
}
```
## Acknowledgements
* We used baseline ResNet-3D code from [3D ResNets for Action Recognition](https://github.com/kenshohara/3D-ResNets-PyTorch)
* We used baseline Hyperbolic embeddings code from [Hyperbolic Action Recognition](https://github.com/Tenglon/hyperbolic_action)
