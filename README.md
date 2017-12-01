# Udacity Self-Driving Car Nanodegree Program 

# **3. Behavioral Cloning** 

**Goals**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image5]: ./images/model_summary.png "Model Image"
[image6]: ./images/center_sample.png "Normal Image"
[image7]: ./images/center_flipped_sample.png "Flipped Image"


### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* behavioral-cloning.py containing the script to create and train the model
* behavioral-cloning.ipynb containing the same script in jupyter notebook
* drive.py for driving the car in autonomous mode (given, with one line added for max speed)
* model.h5 containing a trained convolution neural network 
* README.md (this file) summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python3 drive.py model.h5
```

#### 3. Submission code is usable and readable

The behavioral-cloning.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

Nvidia model has been used, which consists of 5 Conv2D layers with depths between 24, 36, 48, 64, and 64 (function nvidia_model() in behavioral-cloning.py) 

The model uses a RELU layer for each Con2D layer to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting. 

The model was trained and validated on the given dataset driving_log.csv and the associated images in a random 80/20 split. The validation ensured that the model was not overfitting. The model was tested by running it through the simulator. With the limited dataset, the model actually could stay on the track. It rarelly drove to the edges and it could recover when it happened. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road with .20 steering angle corrections from both side images. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Nvidia model has been used, with an additional dropout layer to combat the overfitting. 

The vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final keras model architecture (function nvidia_model() in  behavioral-cloning.py) is summarized in the following:

![alt text][image5]

#### 3. Creation of the Training Set & Training Process

I used the given dataset only to train and validate the self-driving. Here is an example image of center lane driving:

![alt text][image6]

To augment the dataset, I also flipped images and angles thinking that this would the recovery if it went off the track. For example, here is an image that has then been flipped:

![alt text][image7]

In addition to the center, left, and right images, the augmentation entends the dataset with three flipped images for each line of data. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 21 as evidenced by the tests. I used an adam optimizer so that manually training the learning rate wasn't necessary.

I started off with the default driving speed of 9 mph. The final trained model could actually achieve the max driving speed of 30 mph, as shown in the following video:

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/xP5Y3oW2rjQ/0.jpg)](https://www.youtube.com/watch?v=xP5Y3oW2rjQ)
