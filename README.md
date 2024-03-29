# HIerarchical Prototype Explainer (HIPE)

This repository contains the official implementation of our work [Hierarchical Explanations for Video Action Recognition](https://arxiv.org/pdf/2301.00436.pdf).

**Abstract:** *To interpret deep neural networks, one main approach is to dissect the visual input and find the prototypical parts responsible for the classification. However, existing methods often ignore the hierarchical relationship between these prototypes, and thus can not explain semantic concepts at both higher level (e.g., water sports) and lower level (e.g., swimming). In this paper inspired by human cognition system, we leverage hierarchal information to deal with uncertainty: When we observe water and human activity, but no definitive action it can be recognized as the water sports parent class. Only after observing a person swimming can we definitively refine it to the swimming action. To this end, we propose HIerarchical Prototype Explainer (HIPE) to build hierarchical relations between prototypes and classes. HIPE enables a reasoning process for video action classification by dissecting the input video frames on multiple levels of the class hierarchy, our method is also applicable to other video tasks. The faithfulness of our method is verified by reducing accuracy-explainability trade off on ActivityNet and UCF-101 while providing multi-level explanations.* 

**Overview of HIPE:**

<img src="https://github.com/sadafgulshad1/Hierarchical-ProtoPNet/blob/main/Architecture_HProtoPNet.png"  />

## Visual Examples 
**Single Level** 

![alt-text-1](https://github.com/sadafgulshad1/Hierarchical-ProtoPNet/blob/main/sample/original_gif100.gif ) ![alt-text-2](https://github.com/sadafgulshad1/Hierarchical-ProtoPNet/blob/main/sample/most_highly_activated_patch_in_original_img_by_top-3_prototype.gif  )![alt-text-3](https://github.com/sadafgulshad1/Hierarchical-ProtoPNet/blob/main/sample/prototype_activation_map_by_top-3_prototype.gif )  ![alt-text-4](https://github.com/sadafgulshad1/Hierarchical-ProtoPNet/blob/main/sample/top-3_activated_prototype_self_act.gif ) ![alt-text-5](https://github.com/sadafgulshad1/Hierarchical-ProtoPNet/blob/main/sample/top-3_activated_prototype.gif )

*Leftmost: Original Video. Second: Parts in the original video that are highly activated by the prototype. Third:  Saliency map in the original video that are highly activated by the prototype. Fourth: Training videos where prototypes come from. Rightmost: Prototypes.*

**Hierarchical** 

*Grandparent Level:* Human-object interaction

![alt-text-1](https://github.com/sadafgulshad1/Hierarchical-ProtoPNet/blob/main/sample/original_gif1121.gif ) ![alt-text-2](https://github.com/sadafgulshad1/Hierarchical-ProtoPNet/blob/main/sample/most_highly_activated_patch_in_original_img_by_top-6_prototype.gif )![alt-text-3](https://github.com/sadafgulshad1/Hierarchical-ProtoPNet/blob/main/sample/prototype_activation_map_by_top-6_prototype.gif )  ![alt-text-4](https://github.com/sadafgulshad1/Hierarchical-ProtoPNet/blob/main/sample/top-6_activated_prototype_self_act.gif ) ![alt-text-5](https://github.com/sadafgulshad1/Hierarchical-ProtoPNet/blob/main/sample/top-6_activated_prototype.gif )

*Parent Level:* Self-grooming

![alt-text-1](https://github.com/sadafgulshad1/Hierarchical-ProtoPNet/blob/main/sample/original_gif1121.gif ) ![alt-text-2](https://github.com/sadafgulshad1/Hierarchical-ProtoPNet/blob/main/sample/most_highly_activated_patch_in_original_img_by_top-9_prototype.gif  )![alt-text-3](https://github.com/sadafgulshad1/Hierarchical-ProtoPNet/blob/main/sample/prototype_activation_map_by_top-9_prototype.gif )  ![alt-text-4](https://github.com/sadafgulshad1/Hierarchical-ProtoPNet/blob/main/sample/top-9_activated_prototype_self_act.gif ) ![alt-text-5](https://github.com/sadafgulshad1/Hierarchical-ProtoPNet/blob/main/sample/top-9_activated_prototype.gif )

*Child Level:* Blow dry

![alt-text-1](https://github.com/sadafgulshad1/Hierarchical-ProtoPNet/blob/main/sample/original_gif1121.gif ) ![alt-text-2](https://github.com/sadafgulshad1/Hierarchical-ProtoPNet/blob/main/sample/most_highly_activated_patch_in_original_img_by_top-3_prototype_1.gif  )![alt-text-3](https://github.com/sadafgulshad1/Hierarchical-ProtoPNet/blob/main/sample/prototype_activation_map_by_top-3_prototype_1.gif )  ![alt-text-4](https://github.com/sadafgulshad1/Hierarchical-ProtoPNet/blob/main/sample/top-3_activated_prototype_self_act_1.gif ) ![alt-text-5](https://github.com/sadafgulshad1/Hierarchical-ProtoPNet/blob/main/sample/top-3_activated_prototype_1.gif )

*Leftmost: Original Video. Second: Parts in the original video that are highly activated by the prototype. Third:  Saliency map in the original video that are highly activated by the prototype. Fourth: Training videos where prototypes come from. Rightmost: Prototypes.*

## Dataset Preparation
### UCF-101
* Download videos and train/test splits [here](http://crcv.ucf.edu/data/UCF101.php).
* Convert from avi to jpg files using ```util_scripts/generate_video_jpgs.py```

```bash
python -m util_scripts.generate_video_jpgs avi_video_dir_path jpg_video_dir_path ucf101
```

* Generate annotation file in json format similar to ActivityNet using ```util_scripts/ucf101_json.py```
  * ```annotation_dir_path``` includes classInd.txt, trainlist0{1, 2, 3}.txt, testlist0{1, 2, 3}.txt

```bash
python -m util_scripts.ucf101_json annotation_dir_path jpg_video_dir_path dst_json_path
```
### Hierarchical UCF-101
* We define hierarchy for UCF-101 with the number of classes at level one, two, and three being 5, 20, and 101 respectively. The classes at the third level of the hierarchy are the 101 original classes of the dataset. The full hierarchy is included in the file `` UCF-101_hierarchy.csv `` 

## Pre-trained Models
### ResNet-3D Backbone
Pre-trained 3D-ResNet models are available [here](https://drive.google.com/open?id=1xbYbZ7rpyjftI_KCk6YuL-XrfQDz7Yd4). We used ```r3d18_K_200ep.pth ``` trained on kinetics 700 (K) and finetuned it on UCF-101 in our experiments.
### Hierarchical Action Embeddings
We computed the hierarchical action embeddings for the hierarchy we define for UCF-101 in `` UCF-101_hierarchy.csv `` following [Teng.et.al](https://openaccess.thecvf.com/content_CVPR_2020/papers/Long_Searching_for_Actions_on_the_Hyperbole_CVPR_2020_paper.pdf). The precomputed hyperbolic action embeddings are uploaded in the file ``UCF101_two_level_emb.pth`` 

## Running the Code 
* For training the model 
```bash
python main.py --root_path ~/data --video_path ~/UCF-101-JPEG --annotation_path ucf101_01.json \
--result_path results --dataset ucf101 --model resnet \
--model_depth 18 --n_classes 101 --batch_size 128 --n_threads 4 --checkpoint 5
```

* Continue Training from epoch 101. (results/save_100.pth is loaded.)

```bash
python main.py --root_path ~/data --video_path ~/UCF-101-JPEG --annotation_path ucf101_01.json \
--dataset ucf101 --resume_path results/save_100.pth \
--model_depth 18 --n_classes 101 --batch_size 128 --n_threads 4 --checkpoint 5
```
* Calculate top-5 class probabilities of each video using a trained model (results/save_200.pth.)  
Note that ```inference_batch_size``` should be small because actual batch size is calculated by ```inference_batch_size * (n_video_frames / inference_stride)```.

```bash
python main.py --root_path ~/data --video_path ~/UCF-101-JPEG  --annotation_path ucf101_01.json \
--result_path results --dataset ucf101 --resume_path results/save_200.pth \
--model_depth 18 --n_classes 101 --n_threads 4 --no_train --no_val --inference --output_topk 5 --inference_batch_size 1
```

* Perform Inference/validation by calculating top-1 video accuracy of a recognition result (/results/val.json). Note that this is the video level accuracy. For some datasets video level and clip level accuracies vary a lot.

```bash
python -m util_scripts.eval_accuracy ucf101_01.json /results/val.json --subset val -k 1 --ignore
```
### Qualitative Analysis
During the training prototypes for each class are stored in the ``img`` folder at each push epoch. 

* Run ``python local_analysis.py`` to find closest prototypes to the test images at children level.

* Run ``python local_analysis_parents.py`` to find closest prototypes to the test images at ancestor levels.
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
* We adapted ResNet-3D code from [3D ResNets for Action Recognition](https://github.com/kenshohara/3D-ResNets-PyTorch).
* We adapted hyperbolic embeddings code from [Hyperbolic Action Recognition](https://github.com/Tenglon/hyperbolic_action).
* We adapted baseline ProtoPNet code from [ProtoPNet](https://github.com/cfchen-duke/ProtoPNet).
