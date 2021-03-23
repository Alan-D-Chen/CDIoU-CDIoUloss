# CDIoU-CDIoUloss
CDIoU and CDIoU loss is like a convenient plug-in that can be used in multiple models. CDIoU and CDIoU loss have different excellent performances in several models such as Faster R-CNN, YOLOv4, RetinaNet and . There is a maximum AP improvement of 1.9% and an average AP of 0.8% improvement on MS COCO dataset, compared to traditional evaluation-feedback modules. Here we just use as an example to illustrate the code.


## Control Distance IoU and Control Distance IoU Loss Function 



by [Chen Dong](https://github.com/Alan-D-Chen), Miao Duoqian



### Introduction 

Numerous improvements for feedback mechanisms have contributed to the great progress in object detection. In* this paper, we first present an evaluation-feedback module, which is proposed to consist of evaluation system and feedback mechanism. Then we analyze and summarize the disadvantages and improvements of traditional evaluation-feedback module. Finally, we focus on both the evaluation system and the feedback mechanism, and propose  **Control Distance IoU** *and* **Control Distance IoU loss function**  (or **CDIoU** *and* **CDIoU loss** *for short) without increasing parameters or FLOPs in models, which show different significant enhancements on several classical and emerging models. Some experiments and comparative tests show that coordinated evaluation-feedback module can effectively improve model performance. CDIoU and CDIoU loss have different excellent performances in several models such as Faster R-CNN, YOLOv4, RetinaNet and ATSS. There is a maximum AP improvement of 1.9% and an average AP of 0.8% improvement on MS COCO dataset, compared to traditional evaluation-feedback modules.*

There are some potential defects in the current mainstream target detection

* It relies too much on the deepening of the backbone to extract features, so as to improve the accuracy of target detection;

* The deepening of neural network, especially the deepening of backbone and neck, results in huge parameters and flops of the model;

* Compared with the evaluation system (IoUs, the common ones are IoU and GIoU). At present, some new model optimization focuses more on the feedback mechanism (IoU losses), such as IoU loss, smooth loss, GIoU loss,CIoU loss, DIoU loss.



We propose Control Distance IoU and Control Distance IoU Loss Function (**CDIoU** and **CDIoU loss** for short).



###  **Analysis of traditional IoUs and loss functions**

* **Analysis of traditional IoUs**

* **IoU: Smooth L1 Loss and IoU Loss**

* **GIoU and GIoU Loss**

* **DIoU loss and CIoU Loss**

For more information, see ***[Control Distance IoU and Control Distance IoU Loss Function for Better Bounding Box Regression](https://arxiv.org/abs/2103.11696)***



### Installation

**CDIoU** *and* **CDIoU loss** is like a convenient plug-in that can be used in multiple models. CDIoU and CDIoU loss have different excellent performances in several models such as ***Faster R-CNN, YOLOv4, RetinaNet*** and ***[ATSS](https://github.com/sfzhang15/ATSS)***. There is a maximum AP improvement of 1.9% and an average AP of 0.8% improvement on MS COCO dataset, compared to traditional evaluation-feedback modules. Here we just use  ***[ATSS](https://github.com/sfzhang15/ATSS)*** as an example to illustrate the code.

- [ ] *Faster R-CNN,YOLOv4,RetinaNet-R101 (1024),ResNet-50 + NAS-FPN (1280@384),[Detectron2 Mask R-CNN R101-FPN](https://github.com/facebookresearch/detectron2),Cascade R-CNN and FCOS.*

These models use different frameworks, and some even have versions, so no code is provided in this article.

------------------------------------------------------------------------------------------------------------------------------------------------------

This ATSS implementation is based on [FCOS](https://github.com/tianzhi0549/FCOS) and [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and the installation is the same as them. Please check [INSTALL.md](https://github.com/sfzhang15/ATSS/blob/master/INSTALL.md) for installation instructions.

ATSS bridges the gap between anchor-based and anchor-free detection via adaptive training sample selection. Comparison tests on ATSS exclude the essential interference between anchor-based and anchor-free detection. In these tests, the interference of positive and negative sample generation is eliminated, which give tests based on ATSS more representativeness.

### ***CDIoU and CDIoU loss functions***



<img src="https://github.com/Alan-D-Chen/CDIoU-CDIoUloss/blob/main/pics/CDIoU.png" alt="CDIoU" style="zoom: 25%;" />

<img src="https://github.com/Alan-D-Chen/CDIoU-CDIoUloss/blob/main/pics/turning.png" alt="Turning" style="zoom:25%;" />

For more information, see ***[Control Distance IoU and Control Distance IoU Loss Function for Better Bounding Box Regression](https://arxiv.org/abs/2103.11696)***



### **Experiments**

In order to verify the effectiveness of CDIoU and CDIoU loss in object detection, experiments are designed and applied to numerous models in this paper. These models encompass existing classical models and emerging models, reflecting certain robustness and wide adaptability.

![image-20210315210941666](https://github.com/Alan-D-Chen/CDIoU-CDIoUloss/blob/main/pics/image-20210315210941666.png)

<img src="https://github.com/Alan-D-Chen/CDIoU-CDIoUloss/blob/main/pics/image-20210315214303929.png" alt="image-20210315214303929" style="zoom: 67%;" />

For more information, see ***[Control Distance IoU and Control Distance IoU Loss Function for Better Bounding Box Regression](https://arxiv.org/abs/2103.11696)***

## Models

For your convenience, we provide the following trained models. All models are trained with 16 images in a mini-batch and frozen batch normalization (i.e., consistent with models in [FCOS](https://github.com/tianzhi0549/FCOS) and [maskrcnn_benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)).



| Model | Multi-scale |      |      | AP (val) | AP (test-dev) | pth  |
| :---: | :---------: | ---- | ---- | :------: | :-----------: | :--: |
|       |             |      |      |          |               |      |
|       |             |      |      |          |               |      |
|       |             |      |      |          |               |      |
|       |             |      |      |          |               |      |

[1] *The testing time is taken from [FCOS](https://github.com/tianzhi0549/FCOS), because our method only redefines positive and negative training samples without incurring any additional overhead.*
[2] *1x and 2x mean the model is trained for 90K and 180K iterations, respectively.*
[3] *All results are obtained with a single model and without any test time data augmentation such as multi-scale, flipping and etc..*
[4] *`dcnv2` denotes deformable convolutional networks v2. Note that for ResNet based models, we apply deformable convolutions from stage c3 to c5 in backbones. For ResNeXt based models, only stage c4 and c5 use deformable convolutions. All models use deformable convolutions in the last layer of detector towers.*
[5] *The model `ATSS_dcnv2_X_101_64x4d_FPN_2x` with multi-scale testing achieves 50.7% in AP on COCO test-dev. Please use `TEST.BBOX_AUG.ENABLED True` to enable multi-scale testing.*





### MSCOCO test-dev

<img src="https://github.com/Alan-D-Chen/CDIoU-CDIoUloss/blob/main/pics/image-20210315222453294.png" alt="image-20210315222453294" style="zoom:80%;" />

* [**TEST-DEV的竞赛**](https://competitions.codalab.org/competitions/20794#participate)



###  **Tips to improve performances**

* **Floating learning rate**


It is a consensus that the learning rate decreases as the iterative process in the experiment. Further, this paper proposes to check the loss every K iterations and increase the learning rate slightly, if the loss function does not decrease continuously. In this way, the learning rate will decrease
and float appropriately at regular intervals to promote the decrease of the loss function.

* **Automatic GT clustering analysis**

It is well known that AP can be effectively improved by performing cluster analysis on GT in the original dataset. We adjust anchor sizes and aspect ratios parameters based on the results of this cluster analysis. However, we do not know the number of clusters through the current approach. The main solution is to keep trying the number of clusters  *N* , and then judge by the final result AP. Obviously, this exhaustive method takes a lot of time.



## Contributing to the project

Any pull requests or issues are welcome.





## CItations

Please cite our paper in your publications if it helps your research:
And is not true！！
```
@inproceedings{chen2021CDIoU,
  title     =  {Control Distance IoU and Control Distance IoU Loss Function for Better Bounding Box Regression},
  author    =  {Chendong, Miaoduoqian.},
  booktitle =  {ICCV},
  year      =  {2021}
}
```

