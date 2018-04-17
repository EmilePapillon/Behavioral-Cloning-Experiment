# **Behavioral Cloning** 

## Writeup 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following: 
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/cnn-architecture-624x890.png "Model Visualization"
[image2]: ./images/overfitting.png "Overfitting"
[image3]: ./images/loss.png "Normal Loss"
[image4]: ./images/center.png "Center Lane"
[image5]: ./images/model.png "Model"
[image6]: ./images/normal.png "Normal Image"
[image7]: ./images/flipped.png "Flipped Image"
[image8]: ./images/grayscale.png "Grayscaled Image"
[image9]: ./images/translation.png "Translated Image"
[image10]: ./images/brightness.png "Brightness Image"
[image11]: ./images/batch_example.png "Batch"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results
* video.mp4 video showing the car driving on track 1

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model was inspired from [NVIDIA's whitepaper](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) except for some ELU activation layers and dropout layers.

The model includes ELU layers to introduce nonlinearity (code lines 142,143,144,146,147), and the data is normalized in the model using a Keras lambda layer (code line 141). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 145 & 148). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 128). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track (see video).

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 155).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and random translation of the images. To modify the angles for left and right images I used the approximation A = tan A, so if we aim at 10 meters in front of the vehicle the angle can be approximated to be 1/10 for each offset meter. The distance of the right and left camera was approximated to be 1.2, so that A = 1.2/10.  

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to take the model elaborated by NVIDIA. I thought this model was appropriate because it has successfully been used for a similar pupoose. To this model, I added the necessary dropout layers to prevent overfitting because we are working with less data. Also, fully connected layer's depth was choosen empirically and we added a preprocessing layer (normalization and cropping). 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

Initial overfitting problem : 

![alt text][image2]

Solved the overfitting problem :

![alt text][image3]

Note that in these images validation and training losses seem to have been inverted, probably due to a typo in the code. This doesn't prevent the information to be useful. 

To combat the overfitting, I modified the model by adding dropout layer, The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. At this point, I had to create a complex pipeline for augmenting the dataset. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 139-152) consisted of 3 convolution neural networks with the following layers and layer sizes ...

1. Preprocessing layers : 
1.1 cropping 70px from top and 25px from bottom
1.2 Normalization px=(px-128)/128
2. 3x Convolutional layers with kernel = 5x5, depths = 24,36,48, stride = 2 and activation : elu
3. Dropout (50%) 
4. 2x Convolutional layers with kernel = 3x3, depths = 64, 64, stride = 1 and activation : elu
5. Droupout (50%) 
6. Flatten
7. 2x Fully connected : 120, 84
8. Output layer (single output) 

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image5]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving, and two laps in the opposite direction. Here is an example image of center lane driving:

![alt text][image4]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover when it tends to drive off the road. However simply gathering this data did not work and the car was still driving off after the two first curves of the first track. 

To solve this problem, I decided to make this algorithm for data augmentation : 
* for each image paths in the dataset (driving_log.csv) :
* Append a list of unapplied methods
* during training, pick randomly a method from that list and apply it to the image
* Append modified image to the batch
* Continue until all methods have been depleted from all images.

Doing so made it easy for me to create as many augmentation methods as my imagination allowed. The ones I selected were: 
* Grayscaling : for the model to learn color-independent characteristics. The images were then converted to RGB colorspace to mainain a depth of 3. 
* Random translations : x and y translation ranging from 0-64 pixels. Padding was used to keep image size constant. Angle were calculated based on the number of pixels translated in the x axis.
* Mirroring : Angle were calculated by taking the negative of the raw angle.
* Changing randomly the brightness for model to learn not to consider shadows.


I tried recording data points on track 2 but this prevented my model to control the vehicle correctly on track 1. In this sense, the model was not able to generalize so well. To complete the assignment I had to re-train model using exclusively data from track 1. 

To augment the data sat, I also flipped images and angles thinking that this would help reduce the bias to the left that is a characteristic of the track For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

The following images shows a grayscaled image and a randomly translated one and one with a modified brightness
![alt text][image8]
![alt text][image9]
![alt text][image10]

After the collection process, I had 12639 number of data points (37917 images). I then preprocessed this data by applying 4 augmentation filters which gave the model 151668 image/angle pairs to train on. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by the figure above showing the evolution of the validation loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.

Here is an example of a training batch with the labels (angles) shown on the images :

![alt text][image11]
