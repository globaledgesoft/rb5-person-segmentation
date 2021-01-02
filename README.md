# RB5 - Person Segmentation & Background Blur
## Introduction 
A problem statement on the robotics platform, where Qualcomm's Robotics platform (i.e. RB5) device has chosen as a target platform for running the application of Person Segmentation which will perform the Segmentation operation using TFLite library and GlobalEdge's trained Segmentation network. End user application will perform segmentation and blur the background of image except Person in the image.


## Prerequisites
 - flash latest image of RB5 as instructed in Thundercomm documentations
 - setting up the Tensorflow on host system for model conversion
 - Installing TFLite Runtime Library on the RB5. Guide: https://www.tensorflow.org/lite/guide/python
 - Installing Opencv-Python on the RB5.


## Model Conversion
### Converting Tensorflow model in TF Lite 
follow the steps given in url mention below for converting Tensorflow models to the TF Lite :
https://www.tensorflow.org/api_docs/python/tf/compat/v1/lite/TFLiteConverter




## Running the application on RB5

Make sure you are running this commands on the RB5 shell.

 1. Clone the gitlab repository using command given below:
```sh
user@user:~/ $ git clone <GIT URL OF THIS PROJECT>

```

 2. Go to the directory of project root

 ```sh
user@user:~/ $ cd RB5-Person-Segmentation
 ```

 3. To run the source code enter the following commands, Make sure You have installed all dependencies mentioned above in pre-requisite.

 ```sh
 user@user:~/ $ python3 main.py --model model/train_g.tflite
```

