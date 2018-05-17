# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/vf_visualization_data_augmented.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/vf_visualization_data_original.png "Augmented with rotation"
[image4]: ./examples/family_crossing.jpg "Traffic Sign 1"
[image5]: ./examples/right_turn.jpg "Traffic Sign 2"
[image6]: ./examples/speed_30.jpg "Traffic Sign 3"
[image7]: ./examples/stop.jpg "Traffic Sign 4"
[image8]: ./examples/truck.jpg "Traffic Sign 5"
[image9]: ./examples/SoftMax_Probs.png "SoftMax Probabilities"
[imange10]: ./examples/normal_image.png "Color"
[imange11]: ./examples/gray_image.png "Gray"
[imange12]: ./examples/norm_gray_image.png "Normalized Gray"
[imange13]: ./examples/rotate_norm_gray_image.png "Normalized Gray"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 

You're reading it! and here is a link to my [project code](https://github.com/avernethy/CarND-Traffic-Sign-Classifier-Project/tree/develop/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32x32x3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data frequency

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I ran the base classifier directly from the LeNet solution to understand the starting point before making any changes.  Before I could do this I had to run the minimum normalization since it is required for the model to run.  I started playing with the EPOCH, batch sizes, and rates to see how far I could stretch the most basic model and achieved a 0.937 validation with an EPOCH of 10, batch size 64, and rate of 0.002.  Feeling relieved that I could at least make the minimum validation accuracy, I started exploring additional steps.

I then decided to convert the images to grayscale after reading the paper in the link provided in the project for "Traffic Sign Recognition with Mult-Scale Convolutional Networks.  I used a simple cv2 transformation using RGB2GRAY and the validation accuracy went down to about 0.923.  Seems like it went down, so I did a param sweep with EPOCHS, batch sizes and rates just to see if grayscaling had an effect.  It did not have a big effect and I ended up settling for the same parameters.

After doing more reading about other solutions, I came upon using cv2.equalizeHist from an Udacity blog (cited the code).  I decided to try it and achieved about 0.93 validation accuracy reliably.

Going back to the paper, it seem rotation and squashing were the best preprocessing for the images.  I decided to try rotation but after some experimentation, went with using a two random rotations ranging -5 to +5 degrees of rotation per image that I determined was underrepresented in the data set. I used an arbitrary cut off of 500 images in a class to determine representation. I then augmented the data so that the rotated images boosted the counts where there was enough data. Here is a histogram of the augmented data set:

![alt text][image3]

The result of adding the two random rotations was that the validation accuracy went up to about a 0.95 level.  I tried running the augmentation a few times to see how the random rotations affected the validation accuracy.  There was some fluctuation but none dipped below the 0.93 mark.

I was also planning to implement the squash but has spent too much time with testing the rotation, so settled on my 0.95 result and decided to move on to the model architecture

Example of an 'underrepresented' image
![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 24x24x10 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 12x12x10 				|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 8x8x24 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x24 					|
| Fully Connected		|Input 448, output 240							|
| RELU					|												|
| dropout				|Training keep rate= 0.5						|
| Fully Connected		|Input 240, output 84							|
| RELU					|												|
| dropout				|Training keep rate= 0.5						|
| Fully Connected		|Input 84, output 43							|
| RELU					|												|
| Softmax				| etc.        									|
|						|												|
|						|												|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the AdamOptimer as came on the LeNet solution.  As far as the batch size, epochs, and learning rate, I periodically tested different ones as I was modifying the model as described in the section below.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.960
* validation set accuracy of 0.990
* test set accuracy of 60% on the five images

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

Itertive Solution I Took:
 I started with the basic LeNet Classifier.  Since this is the first time I have played with a ConvNets I took it one modification at a time to see the effect of different stages.
 
 The first thing I did was add the dropouts.  First, on the fully connected layer before the logits and the the layer above. Using a fresh rotation, grayscaled, normalized set, the validation accuracy was 0.946, moved to 0.950 after the first dropout was added, and again to 0.948.  However, I noticed that the training accuracy had gone down from about 0.99 to 0.98 or lower which I thought meant the model might be less susceptible to overfitting.
 
Next I wanted to try to play with the depth of the model since I saw in the lecture that adding more layers and getting the model to be more deep could improve accuracy.  I started with playing with the fully connected layers and settled changing the 2nd fully connected layer from the default of 120 to 240. This yielded a validation accuracy of 0.957.  Slight improvement so I decided to try adding another convolution layer.  I also tried different EPOCHS, rate and batch size.  I found that EPOCHS = 10, rate = 0.001 and batch size had decreased the training time but not affected the validation accuracy much so stuck with those values

Finally I decided to add a convolution layer (output 24x24x10). Again, the goal was to have the model become more deep but keep the training parameters to be close the original.  Originally, the first fully connected layer had an input of 400 but after choosing the padding style, and number of filters, I ended up with an input of 448 for the first fully connected layer.  This yielded a validation 0.978.  However, I noticed that the validation accuracy was not changing much after 5 EPOCHS so I also decreased the number of EPOCHS from 10 to 5.  This hurt a little in validation accuracy 0.97 to 0.96 but the training accuracy had also come down a bit, so again I figured this would help with the chances of overfitting.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image, Children Crossing, might be difficult to classify because the image of the two people crossing could be lost in the downsampling.  The second image, Turn Right Ahead, may be hard to classify since the background color of the sign closely matches the color of the sky, and the shape of the arrow might get missclassified.  The third image, 30km/hr, might have difficulty with the number 3 and 8.  The fourth image, Stop, may be difficult because of the angle at which the photo is taken.  The fifth, Vehicles Over 3.5 Metric Tons Prohibited, image may have trouble because of the truck shape

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Children crossing  	| Children crossing				  		 		| 
| Turn right ahead  	| Ahead only        							|
| Speed limit (30km/h)	| Speed limit (30km/h)					 		|
| Stop					| Priority road									|
| Vehicles over 3.5 tons| Vehicles over 3.5 metric tons prohibited     	|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This is in contrast to the test set accuracy of 93.7%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .47         			| Children crossing								| 
| .56     				| Ahead only        							|
| .89					| Speed limit (30km/h)							|
| .54	      			| Priority road					 				|
| .92				    | Vehicles over 3.5 metrics tons prohibited		|

![alt text][image9]

The speed limit and 3.5 metric ton vehicle signs had very high probability while the other three images were only a roughly 50/50 chance of being correct. It is surprising that the Children crossing was predicted correctly given the probability was less than 50%. 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


