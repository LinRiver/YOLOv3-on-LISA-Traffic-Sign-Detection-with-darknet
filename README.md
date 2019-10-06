# YOLOv3-on-LISA-Traffic-Sign-Detection-with-darknet
This project is to improve YOLOv3 performance by GIOU instead of IOU and the integration of conv and batch_normalization layers

# Dependence
1. Ubuntu 16.04
2. darknet
3. Python 3.5

# Dataset
LISA Traffic Sign Dataset: Laboratory for Intelligent and Safe Automobiles
Link: http://cvrr.ucsd.edu/LISA/lisa-traffic-sign-dataset.html

# Implementation
1. Download darknet framework
Link: https://github.com/pjreddie/darknet
2. Download pretrained model weight
Link: https://pjreddie.com/media/files/darknet53.conv.74
3. Convert the data format of LISA dataset by pip install Jinja2-2.10.1-py2.py3-none-any.whl and parse_lisa.py on LISA_TS/allAnnotations.csv
4. Add data/voc-lisa.names
5. Add cfg/voc-lisa.data
6. Add cfg/yolov3-voc-lisa-giou.cfg with arg set up
7. For the integration of conv and batch_normalization layers, define fuse_conv_batchnorm(network net) in include/darknet.h and src/network.c, and call it in src/demo.c and examples/detector.c
8. For given the choice between optimizing a metric itself vs. a surrogate loss function, the
optimal choice is the metric itself. The usage of GIOU insteads of l-1 and l-2 norms
9. ./darknet detector train cfg/voc-lisa.data cfg/yolov3-voc-lisa-giou.cfg darknet53.conv.74 2>1

# TODO
1. Train model with LISA by CuDNN and OpenCV
2. The visualization of training performance
3. The usage of Clustering algorithms on Anchor Box size
4. Compress YOLOv3
5. Convert trained model to Caffe model for the deployment of Embedded system

# Reference
1. Darknet: Open Source Neural Networks in C, https://pjreddie.com/darknet/
2. Generalized Intersection over Union: a Metric and a Loss for Bounding Box Regression, https://arxiv.org/pdf/1902.09630.pdf




