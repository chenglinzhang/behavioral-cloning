# Udacity Self-Driving Car Nanodegree Program 

# **2. Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/30miles.png "30 miles sign"
[image5]: ./examples/histogram.png "train data histogram"
[image6]: ./examples/13-1.jpg "traffic sign"
[image7]: ./examples/36-1.jpg "traffic sign"
[image8]: ./examples/0-1.jpg "traffic sign"
[image9]: ./examples/18-1.jpg "traffic sign"
[image10]: ./examples/38-1.jpg "traffic sign"

---
### Writeup / README

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

I use matplotlib to visualize images in the traffic signs dataset. An example of 30 miles sign is shown in the following:  

![alt text][image4]

The train dataset is shown by an histogram in matplotlib. It is a bar chart showing how the data is distributed among the class labels.

![alt text][image5]

The distribution is unbalanced, as is the case for most real-world classification problems.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I normalized the image data because it centers the data and makes the range similar in order to get stable gradients, which generally helps classifier to converge faster.

The train dataset, for example, is normalized by its mean and sigma in the following:

<pre>
    X_train = (X_train - np.mean(X_train))/np.std(X_train, axis=0)
</pre>

I decided not to convert the images to grayscale because the convertion loses color information, which can be critical to the recognition of certain signs.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Flattern layer	    | outputs 400  									|
| Fully connected		| 400 to 120    								|
| Fully connected		| 120 to 84    		    						|
| Output layer  		| 84 to 43    		    						|
| Softmax				|           									|
|						|												|

It is a slight modification to LeNet for 3-channel color images.


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used 

* Adam Optimizer
* Epochs 150
* Batch size 128
* Learning rate 0.001

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.933
* test set accuracy of 0.926

The validation accuracy and test accuracy are very close, which provides evidence that the model is trained well.

The LeNet architecture was chosen due to its simplicity. LeNet was originally designed for gray-scale image inputs. I want to see how far it can be applied for color images. My first run sets the validation and test accuracies close to 90%, which made me believe it is general enough for traffic sign recognition. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I used 18 German traffic signs to verify the predictions. Five of them are shown in the following

![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10]

The classifier does not perform well on speed limit signs. It tends to mistake the numbers inside the circle.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The model was able to correctly guess 15 of the 18 traffic signs, which gives an accuracy of 83.3%. This compares favorably to the accuracy on the test set of 92.6%. 

Here are the results of the prediction for the 5 images among the 18:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Go straight or left	| Go straight or left   						| 
| Yield					| Yield											|
| 20 km/h	      		| 60 km/h (WRONG)				 				|
| General Caution	 	| General Caution      			 				|
| Keep Right      		| Keep Right               						|

The third one was recognized wrong because 20km/h and 60km/h images match in major areas. The classifier does not have "attention" mechanism to tell the differences between 2 and 6.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The model is relatively sure for each prediction. The third one is wrong though it got the probability of 1.0.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00	                | Go straight or left   						| 
| 0.99					| Yield											|
| 1.00	      	    	| 60 km/h (WRONG)				 				|
| 1.00	 	            | General Caution      			 				|
| 1.00      	    	| Keep Right               						|


# behavioral-cloning
