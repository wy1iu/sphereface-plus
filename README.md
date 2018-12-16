# *SphereFace+* : Improving Inter-class Feature Separability via MHE for Face Recognition


### License

SphereFace+ is released under the MIT License (refer to the LICENSE file for details).

### Content
1. [Introduction](#introduction)
2. [Citation](#citation)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Results](#results)
7. [Notes](#notes)
7. [Reference](#reference)
8. [Contact](#contact)


### Introduction
Inspired by prior knowledge that weights of classifier represent the center of each class respectively, we propose SphereFace+ by applying Minimum Hyperspherical Energy (MHE), which can effectively enhance inter-class feature separability, to [SphereFace](https://github.com/wy1iu/sphereface). Our experiments verify MHE's abilities of improving inter-class feature separability and further boosting the performance of SphereFace for face recognition. Our paper is available at [arXiv](https://arxiv.org/abs/1805.09298) (SphereFace+ is described in Section 5.2 of the main paper).


### Citation
If you find **SphereFace+** useful in your research, please consider to cite the following paper:

	  @InProceedings{LiuNIPS18,
             title={Learning towards Minimum Hyperspherical Energy},
             author={Liu, Weiyang and Lin, Rongmei and Liu, Zhen and Liu, Lixin and Yu, Zhiding and Dai, Bo and Song, Le},
             booktitle={NIPS},
             year={2018}
	  }

and the original **SphereFace**:

	  @InProceedings{Liu2017CVPR,
             title = {SphereFace: Deep Hypersphere Embedding for Face Recognition},
             author = {Liu, Weiyang and Wen, Yandong and Yu, Zhiding and Li, Ming and Raj, Bhiksha and Song, Le},
             booktitle = {CVPR},
             year = {2017}
	  }

### Requirements
1. Requirements for [CUDA 8.0 for Linux](https://developer.nvidia.com/cuda-80-ga2-download-archive)
2. Requirements for [cuDNN v6.0 (April 27, 2017), for CUDA 8.0](https://developer.nvidia.com/rdp/cudnn-archive) (**Important!!!**)
3. Requirements for `Matlab`
4. Requirements for `Caffe` and `matcaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))
5. Requirements for `MTCNN` (see: [MTCNN - face detection & alignment](https://github.com/kpzhang93/MTCNN_face_detection_alignment)) and `Pdollar toolbox` (see: [Piotr's Image & Video Matlab Toolbox](https://github.com/pdollar/toolbox)).

**Attension:** If you used other CUDA or cuDNN versions, the training process would fail frequently.

### Installation
1. Clone recursively the SphereFace-Plus repository. We'll call the directory that you cloned SphereFace-Plus as **`SPHEREFACE_PLUS_ROOT`**. The installation basically follows [SphereFace](https://github.com/wy1iu/sphereface).

2. Build Caffe and matcaffe

    ```Shell
    cd $SPHEREFACE_PLUS_ROOT/tools/caffe-sphereface
    # Now follow the Caffe installation instructions here:
    # http://caffe.berkeleyvision.org/installation.html
    make all -j8 && make matcaffe
    ```

If you have any questions about installation caffe with cudnn 6.0, try to refer to [caffe issue #1325](https://github.com/BVLC/caffe/issues/1325#issuecomment-219810559).


### Usage

*After successfully completing the [installation](#installation)*, you are ready to run all the following experiments.

#### Part 1: Preprocessing

> **the same as** [SphereFace preprocessing](https://github.com/wy1iu/sphereface#user-content-part-1-preprocessing)

**Note:** In this part, we assume you are in the directory **`$SPHEREFACE_PLUS_ROOT/preprocess/`**
1. Download the training set (`CASIA-WebFace`) and test set (`LFW`) and place them in **`data/`**.

	```Shell
	mv /your_path/CASIA_WebFace  data/
	./code/get_lfw.sh
	tar xvf data/lfw.tgz -C data/
	```
    Please make sure that the directory of **`data/`** contains two datasets.
    
2. Detect faces and facial landmarks in CAISA-WebFace and LFW datasets using `MTCNN` (see: [MTCNN - face detection & alignment](https://github.com/kpzhang93/MTCNN_face_detection_alignment)).

	```Matlab
	# In Matlab Command Window
	run code/face_detect_demo.m
	```
    This will create a file `dataList.mat` in the directory of **`result/`**.
3. Align faces to a canonical pose using similarity transformation.

	```Matlab
	# In Matlab Command Window
  	run code/face_align_demo.m
  	```
    This will create two folders (**`CASIA-WebFace-112X96/`** and **`lfw-112X96/`**) in the directory of **`result/`**, containing the aligned face images.


#### Part 2: Train

**Note:** In this part, we assume you are in the directory **`$SPHEREFACE_PLUS_ROOT/train/`**

1. Get a list of training images and labels.

	```Shell&Matlab
	mv ../preprocess/result/CASIA-WebFace-112X96 data/
	# In Matlab Command Window
	run code/get_list.m
	```
    The aligned face images in folder **`CASIA-WebFace-112X96/**` are moved from ***preprocess*** folder to ***train*** folder. A list `CASIA-WebFace-112X96.txt` is created in the directory of **`data/`** for the subsequent training.

2. Get pretrained models from [Google Drive](https://drive.google.com/open?id=1Vn7LRz_c1zU0sJ7-bKDp_O4t_AMQiozL) | [BaiduYunDisk](https://pan.baidu.com/s/19ueWgPHw85UFXDw8G657gw).

    Download all pretrained models from [Google Drive](https://drive.google.com/open?id=1Vn7LRz_c1zU0sJ7-bKDp_O4t_AMQiozL) | [BaiduYunDisk](https://pan.baidu.com/s/19ueWgPHw85UFXDw8G657gw). And move them into **`$SPHEREFACE_PLUS_ROOT/train/pretrained_model/`**. We initialize our network with such pretrained models for computing inter class distances better.

	Pretrained Models |Single|Double|Triple|Quadruple
	:---:|:---:|:---:|:---:|:---:
    ACC|96.22%|98.87%|98.93%|99.27%
	

3. Train the sphereface model.

	1. For m = 4
        ```Shell
	    bash train_sfplus.sh
	    ```
        We use 2 GPUs to run training. If you want to use only one GPU, please set **`iter_size = 2`** in **`code/sfplus/sfplus_solver.prototxt`** and change **`train_sfplus.sh`** manually.
        After training, a model `sfplus_model_iter_8000.caffemodel` and a corresponding log file `sfplus_train.log` are placed in the directory of `result/`.
	
	2. For m = 1
        ```Shell
	    bash train_m_single.sh
	    ```
	3. For m = 2
        ```Shell
	    bash train_m_double.sh
	    ```
	4. For m = 3
        ```Shell
	    bash train_m_triple.sh
	    ```

See more traing detail in [Training Notes](https://github.com/wy1iu/sphereface-plus/blob/master/TrainingNotes.md)

#### Part 3: Test

**Note:** In this part, we assume you are in the directory **`$SPHEREFACE_PLUS_ROOT/test/`**

1. Get the pair list of LFW ([view 2](http://vis-www.cs.umass.edu/lfw/#views)).

	```Shell
	mv ../preprocess/result/lfw-112X96 data/
	./code/get_pairs.sh
	```
	Make sure that the LFW dataset and`pairs.txt` in the directory of **`data/`**

2. Extract deep features and test on LFW.

	```Shell
	matlab -nodisplay -nodesktop -r evaluation
	```
    Finally we get the accuracy on LFW.

**Attention:** You can also test `sfplus_model_iter_7000.caffemodel` by changing **`test/code/evaluation.m`**.



### Results

1. m = 4
    * For m = 4, we go through the entire pipeline for 10 times. The accuracies on LFW are shown below. And we release model #7.

	    Experiment |#1|#2|#3|#4|#5|#6|#7(released)|#8|#9|#10
	    :---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
        ACC|99.23%|99.30%|99.25%|99.28%|99.20%|99.27%|**99.35%**|99.18%|99.33%|99.28%

	* Released Training Log & Model File [Google Drive](https://drive.google.com/open?id=1kpGGvb5Nv0EmDicW2ue8LdUVjI0Ht1PP) | [BaiduYunDisk](https://pan.baidu.com/s/1N2IAIXnhROo9NubonTVbBA)
2. m = 1
    * For m = 1, we go through the entire pipeline for 5 times. The accuracies on LFW are shown below. And we release model #3.

	    Experiment |#1|#2|#3(released)|#4|#5
	    :---:|:---:|:---:|:---:|:---:|:---:
        ACC|97.48%|97.32%|97.48%|97.18%|97.53%

	* Released Training Log & Model File [Google Drive](https://drive.google.com/open?id=18xTKUGRbg0DnuHZJv2yNp2nmzGF5x87G) | [BaiduYunDisk](https://pan.baidu.com/s/10Ht1GF4QsL-rBwjbWHf7KA)
3. m = 2
    * For m = 2, we go through the entire pipeline for 8 times. The accuracies on LFW are shown below. And we release model #3.

	    Experiment |#1|#2|#3(released)|#4|#5|#6|#7|#8
	    :---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
        ACC|98.95%|98.98%|99.05%|99.08%|98.90%|99.02%|99.05%|98.83%

	* Released Training Log & Model File [Google Drive](https://drive.google.com/open?id=1CI4n_e9webaG7D4xPC65MHTo8Blbu7S8) | [BaiduYunDisk](https://pan.baidu.com/s/1f2dZoYU_Xx2n8ytAhPECLA)
4. m = 3
    * For m = 3, we go through the entire pipeline for 8 times. The accuracies on LFW are shown below. And we release model #5.

	    Experiment |#1|#2|#3|#4|#5(released)|#6|#7|#8
	    :---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
        ACC|98.93%|99.05%|99.08%|99.05%|99.08%|98.90%|99.13%|99.00%

	* Released Training Log & Model File [Google Drive](https://drive.google.com/open?id=1bXU8gvSGQzaDbhvtWgtMtYcZhKlRQzZP) | [BaiduYunDisk](https://pan.baidu.com/s/1cjCEO7IQVLZOqO2TSK4jbw)


All models can find in [Google Drive](https://drive.google.com/drive/folders/1mDGdp-BOuawF345P3BQ9Q8Z6KOEPX3fC?usp=sharing) | [BaiduYunDisk](https://pan.baidu.com/s/13KawnSc2i6IWuFzkq1vnhg)

### Notes
1. **Pretraining is a very effective way to avoid training difficulty.**

    As one can learn from our implementation, we use the pretrained model from the original SphereFace, and finetune the SphereFace model using the new loss of SphereFace+. It can effectively reduce the training difficulty of the new loss and improve the results consistently.
    
2. **Finetuning the CASIA-pretrained model on new datasets could potentially stablize the training difficulty.**

    When you are using our model for some new datasets, you can also consider finetuning the CASIA-trained models on the new datasets.



### Reference

1. [caffe-sphereface](https://github.com/wy1iu/sphereface)
2. [caffe-AM-Softmax](https://github.com/happynear/AMSoftmax)

### Contact

   [Lixin Liu](https://liulixinkerry.github.io) and [Weiyang Liu](https://wyliu.com)

  Questions can also be left as issues in the repository. We will be happy to answer them.
