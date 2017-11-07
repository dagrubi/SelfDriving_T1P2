#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./pics/example_1.png "Example 1"
[image2]: ./pics/histogramm.png "Histogramm of training set"
[image3]: ./pics/grayscale.png "grayscaling"
[image4]: ./pics/normalization.png "normalisatzion"
[image5]: ./pics/nr_epochs.png "nr_epochs"
[image6]: ./new_data/baustelle.jpg "Traffic Sign 1"
[image7]: ./new_data/left.jpg "Traffic Sign 2"
[image8]: ./new_data/speed_limit.jpg "Traffic Sign 3"
[image9]: ./new_data/stop.jpg "Traffic Sign 4"
[image10]: ./new_data/vorfahrt_achten.jpg "Traffic Sign 5"
[image11]: ./new_data/traffic_light.jpg "Traffic Sign 6"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/dagrubi/Term1_P2/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. 

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3 
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 
As exmaple four traffic signs are shown. The label can be found in signnames.csv
.
![Example 1][image1]

The brighntess and contrast of the picture is different. On most pictures the traffic sign is in the center of the picture. Due to the various brightness and contrast the training data set is usable for deep learning approaches.

It is a bar chart shows the distirbution of the training data. The xlabel represents the number of classes as defined in signnames.csv.
.
![Histogramm][image2]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

As a first step, I decided to convert the images to grayscale because for the detection of traffic signs color might not be necessary. Less information has to be processed -> training time will be shorter
As example for the visualisatzion i choosed training data nr. 1567, a sign for traffic lights:
.
![Example for grayscaling][image3]

As last step i normalized the image data to values between -1 and 1, since normalized data can be trained faster and the danger of getting stuck in local minimas is less.

Alternatively i normalized the colored images. I want to test which variant is better:
.
![Example for normalized colored image][image4]

I did not use any further preprocessing techniques.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model (LeNet_RGG() in the code!!) consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x24 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x24 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x48 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x48 				|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 3x3x172 	|
| RELU					|												|
| Max pooling	      	| 1x1 stride,  outputs 2x2x172 				|
| Flatten				| inputs	2x2x172 , output 688|
| Fully Connected | connect every neuron	input: 688, output: 400|
| RELU					|												|  
| Dropout | dropout-rate 0.5 |
| Fully Connected | connect every neuron	input: 400, output: 129|
| RELU					|												|
| Dropout | dropout-rate 0.5 |
| Fully Connected | connect every neuron	input: 129, output: 43|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I trained the model on a gpu on AWS (g2.2xlarge).
As optimizer the Adam-Optimizer is used

Batch Size: 128
Nr. of Epochs: 40
learning rate: 0.001
Sigma: 0.1

The following graphs shows the increase of the validation accuracy depending on the number of epochs:
.
![Validation accuracy vs. number of epochs][image5]

After 30 epochs only a small increase of validation accuracy is noticeable.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* test set accuracy of 94.1%
* validation set accuracy of 96.5%  
* train set accuracy of 99.7%

Model development
I choose as basis the LeNet-5 architecture, which was shown in the LeNetLab in the udacity classroom. I modified the network concerning the requested inputs (32x32x3) and the requested outputs (43 labels).
I did several modifications:
- Validation accuracy was low (less than 80%), so i added an additional convolutional layer
- Dropout after the fully connected layers improved the validation accuracy also.
- Dropout after the convoluational layers did not improve the validation accuracy.
- I increased the height of the convolutional layers in order to increase validation accuracy

I tested the changes on the architecture on the gray images and on the normalized rgb-images. I reached an validation accuracy arount 90% for the gray images, so i continued only with the normalized rgb-images.

tuned parameters:
 - epoch size (max. 40 epochs were enough, no improvements after 40 epochs)
 - height of convolutional layers 1,2,3
 - height of fully connected layers

 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![construction][image6] 
![left curve][image7] 
![speed limit][image8] 
![stop][image9] 
![yield][image10] 
![traffic_light][image11]

The traffic light might difficult to classifiy, since it is skewed and there is another sing visable
All other images might easy to classify, since the traffic sign are central and the brightness and colorness is good.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road work     		| Road work   									| 
| Dangerous curve to the left     			| Dangerous curve to the left  										|
| 120 km/h	      		| 120 km/h	  				 				|
| stop			| stop      							|
| Yield					| Yield											|
| Traffic light					| 			30 km/h							|


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83%. This is compareable to the validaton accuracy around 94% in the test data.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.

For the first image is road work
![construction][image6] 
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Road Work   									| 
| 0.00     				| beware of ice 										|
| 0.00					| 70 km/h										|
| 0.00	      			| 20 km/h					 				|
| 0.00				    | 30 km/h     							|


The second image is dangerious curve to left
![dangerious left curve][image7] 
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| dangerious curve to left   									| 
| 0.00     				| Go straight or left										|
| 0.00					| Double curve								|
| 0.00	      			| slippery road			 				|
| 0.00				    | no passing for vehicles > 3.5t  						|

The third image is 120 km/h
![120 km/h][image7] 
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| 120 km/h   									| 
| 0.00     				| 20 km/h 										|
| 0.00					| 30 km/h 								|
| 0.00	      			| 50 km/h 	 				|
| 0.00				    | 60 km/h |			

The 4th image is stop
![stop][image9] 
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| stop 									| 
| 0.00     				| 20 km/h 										|
| 0.00					| 30 km/h 								|
| 0.00	      			| 50 km/h 	 				|
| 0.00				    | 60 km/h |						

The 5th image is yield
![yield][image10] 
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| yield  									| 
| 0.00     				| end of no passing by vehicles > 3.5t									|
| 0.00					| 20 km/h 						|
| 0.00	      			| 30 km/h 	 				|
| 0.00				    | 50 km/h  			|

The 6th image is traffic signals
![traffic signals][image11] 
A speed limit of 30 km/h was predicted (wrong)
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99         			| 30 km/h  									| 
| 0.00     				| General caution								|
| 0.00					| priority road					|
| 0.00	      			| end of no passing	 				|
| 0.00				    | 60 km/h			

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


