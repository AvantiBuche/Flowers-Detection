# Flowers-Detection
# 

#### Flower detection system which will detect objects based on what type of flower.

## Aim and Objectives

### Aim
To create a real-time video flower detection system which will detect objects based on whether it is sunflower, rose, tulip.

### Objectives
➢ The main objective of the project is to create a program which can be either run on Jetson nano or any pc with YOLOv5 installed and start detecting       using the camera module on the device.

➢ Using appropriate datasets for recognizing and interpreting data using machine learning.

➢ To show on the optical view finder of  the camera  module whether objects are sunflower, rose, tulip.

### Abstract

➢ An object is classified based on whether it is  sunflower, rose, tulip. is detected by the live feed from the system’s camera.

➢ We have completed this project on jetson nano which is a very small computational device.

➢ A lot of research is being conducted in the field of Computer Vision and Machine Learning (ML), where machines are trained to identify various objects   from one another. Machine Learning provides various techniques through which various objects can be detected.

➢ One such technique is to use YOLOv5 with Roboflow model, which generates a small size trained model and makes ML integration easier.


### Introduction
➢ This project is based on flower detection model with modifications. We are going to implement this project with Machine Learning and this       project can be even run on jetson nano which we have done.

➢ This project can also be used to gather information about what category of flower object comes in.

➢ The objects can even be further classified into fire and smoke based on the image annotation we give in roboflow.

➢ Flower detection sometimes becomes difficult as certain mixed together and gets harder for the model to detect. However, training in Roboflow     has allowed us to crop images and also change the contrast of certain images to match the time of day for better recognition by the model.

➢ Neural networks and machine learning have been used for these tasks and have obtained good results.

➢ Machine learning algorithms have proven to be very useful in pattern recognition and classification, and hence can be used for flower detection as well.


### Jetson Nano Compatibility

➢ The power of modern AI is now available for makers, learners, and embedded developers everywhere.

➢ NVIDIA® Jetson Nano™ Developer Kit is a small,  powerful computer that lets you run multiple neural networks in parallel for applications like image     classification,  object detection,  segmentation,  and  speech processing. All in an easy-to-use platform that runs in as little as 5 watts.

➢ Hence due to ease of process as well as reduced cost of implementation we have used Jetson nano for model detection and training.

➢ NVIDIA JetPack SDK is the most comprehensive solution for building end-to-end accelerated AI applications. All Jetson modules and developer kits are     supported by JetPack SDK.

➢ In our model we have used JetPack version 4.6 which is the latest production release and supports all Jetson modules.

### Proposed System
    
1. Study basics of machine learning and image recognition.
    
2. Start with implementation

   ➢ Front-end development

   ➢ Back-end development

3. Testing, analysing and improvising the model. An application using python and Roboflow and its machine learning libraries will be using machine  learning to identify whether objects are sunflower, rose, tulip.

4. Use datasets to interpret the object and suggest whether the object is sunflower, rose, tulip.

### Methodology

The Fire and smoke detection system is a program that focuses on implementing real time Fire and smoke detection.

It is a prototype of a new product that comprises of the main module: Fire and smoke detection and then showing on view finder whether the object is Fire and smoke or not.

#### Flower Detection Module

#### This Module is divided into two parts:

#### 1] Flower detection

➢ Ability to detect the location of object in any input image or frame. The output is the bounding box coordinates on the detected object.

➢ For this task, initially the Dataset library Kaggle was considered. But integrating it was a complex task so then we just downloaded the images from     gettyimages.ae and google images and made our own dataset.

➢ This Datasets identifies object in a Bitmap graphic object and returns the bounding box image with annotation of object present in a given image.

#### 2] Classification Detection

➢ Classification of the object based on whether it is sunflower, rose, tulip.

➢ Hence YOLOv5 which is a model library from roboflow for image classification and vision was used.

➢ There are other models as well but YOLOv5 is smaller and generally easier to use in production. Given it is natively implemented in PyTorch (rather       than Darknet), modifying the architecture and exporting and deployment to many environments is straightforward.

➢ YOLOv5 was used to train and test our model for various classes like Fire and smoke. We trained it for 149 epochs and achieved an accuracy of             approximately 91%.

### Jetson Nano 2GB Developer Kit.





![IMG_20220125_115121](https://user-images.githubusercontent.com/100038142/170922938-5e7dd2ca-f4c2-4e2c-b7f9-fa35e331702b.jpg)







### Setup

## Installation

### Initial Setup

Remove unwanted Applications.
```bash
sudo apt-get remove --purge libreoffice*
sudo apt-get remove --purge thunderbird*
```
### Create Swap file

```bash
sudo fallocate -l 10.0G /swapfile1
sudo chmod 600 /swapfile1
sudo mkswap /swapfile1
sudo vim /etc/fstab
```
```bash
#################add line###########
/swapfile1 swap swap defaults 0 0
```
### Cuda Configuration

```bash
vim ~/.bashrc
```
```bash
#############add line #############
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export
LD_LIBRARY_PATh=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_P
ATH}}
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
```
```bash
source ~/.bashrc
```
### Update a System
```bash
sudo apt-get update && sudo apt-get upgrade
```
################pip-21.3.1 setuptools-59.6.0 wheel-0.37.1#############################

```bash 
sudo apt install curl
```
``` bash 
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
```
``` bash
sudo python3 get-pip.py
```
```bash
sudo apt-get install libopenblas-base libopenmpi-dev
```

```bash
vim ~/.bashrc
```

```bash
sudo pip3 install pillow
```
```bash
curl -LO https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl
```
```bash
mv p57jwntv436lfrd78inwl7iml6p13fzh.whl torch-1.8.0-cp36-cp36m-linux_aarch64.whl
```
```bash
sudo pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl
```
```bash
sudo python3 -c "import torch; print(torch.cuda.is_available())"
```
### Installation of torchvision.

```bash
git clone --branch v0.9.1 https://github.com/pytorch/vision torchvision
cd torchvision/
sudo python3 setup.py install
```
### Clone yolov5 Repositories and make it Compatible with Jetson Nano.

```bash
cd
git clone https://github.com/ultralytics/yolov5.git
cd yolov5/
```

``` bash
sudo pip3 install numpy==1.19.4
history
##################### comment torch,PyYAML and torchvision in requirement.txt##################################
sudo pip3 install --ignore-installed PyYAML>=5.3.1
sudo pip3 install -r requirements.txt
sudo python3 detect.py
sudo python3 detect.py --weights yolov5s.pt --source 0
```

## Fire Dataset Training
### We used Google Colab And Roboflow

train your model on colab and download the weights and pass them into yolov5 folder
link of project


## Running Fire Detection Model
source '0' for webcam

```bash
!python detect.py --weights best.pt --img 416 --conf 0.1 --source 0
```

### Demo



https://youtu.be/sQzcbxqt-Wo





### Advantages

➢ Video-based flower detection is currently a standard technology due to image processing, computer vision, and Artificial Intelligence. These systems have   remarkable potential advantages over traditional methods, such as a fast response and wide detection areas.


➢ Deep learning techniques have the advantage of extracting the features automatically, making this process more effective and dramatically improving the   state-of-the-art in Image Classification and object detection methods

➢ It can then convey to the person who present in control room  if it needs to be completely automated 

➢ When completely automated no user input is required and therefore works with absolute efficiency and speed.

➢ It can work around the clock and therefore becomes more cost efficient.

### Application

➢ Detects object class like sunflower, rose, tulip in a given image frame or view finder using a camera module.

➢ Can be used in various places.

➢ Can be used as a refrence for other ai models based on flower detection.

### Future Scope

➢ As we know technology is marching towards automation, so this project is one of the step towards automation.

➢ Thus, for more accurate results it needs to be trained for more images, and for a greater number of epochs.

➢ Flower  segregation will become a necessity in the future due to rise in population and hence our model will be of great help to tackle the       situation in an efficient way.

➢  In future our model which can be trained and modified with just the addition of images can be very useful.

### Conclusion

➢ In this project our model is trying to detect objects and then showing it on view finder, live as what their class is as whether they are  sunflower, rose, tulip or not as we have specified in Roboflow.


### Refrences

#### 1] Roboflow :- https://roboflow.com/

#### 2] Datasets or images used: kaggle

#### 3] Google images

### References

https://sciencing.com/importance-of-flowers-in-nature-12000163.html
