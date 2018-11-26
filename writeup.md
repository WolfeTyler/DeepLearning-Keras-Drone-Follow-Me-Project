## Project: Follow Me


[//]: # (Image References)
[image0]: ./images/FCN.png
[image1]: ./images/hyper_params.png
[image2]: ./images/eval_results.png
[image3]: ./images/train_loss.png
[image4]: ./images/network_design.png




### Network Setup
The FCN network is used to perform inference against the pixels in an image. The FCN is composed of a series of convolution layers down to a 1x1.

The first section of the network is the encoder. The 1x1 convolution layer is not tall or wide but is deep in filters. Although the end of the encoder is 1x1 the output isn't necessarily 1x1.

The next section of the network is the decoder. It's composed of transposed convolutions that increase the height and width while shortening the depth in opposite fashion as the end of the encoder.

To solve the follow me challenge I used a 3 encoder, 1x1 convolution, 3 decoder setup.

![alt_text][image0]

![alt_text][image4]


### Network & Hyper Parameters

The source code contained a series of default hyper parameters:

learning_rate = 0
batch_size = 0
num_epochs = 0
steps_per_epoch = 200
validation_steps = 50
workers = 2

I iteratively adjusted one parameter at a time in order to perform a controlled experiment to understand what postitive/negative impacts the parameter adjustments would have to the results and final model performance scoring.

- Batch_size I updated first to align with the volume of images being trained against
- Num_epochs I adjusted second through trial and error starting with 10, then 20, finally 30 where it appeared the improvement was statistically trailing off through diminishing returns
- Steps_per_epoch & validation_steps I decided to leave as is
- Workers I changed from 2 down to 1 to keep simple
- Learning_rate I knew from prior experience is very important to model performance. Initially I started with a smaller learning_rate of .00001, however as somewhat expected the val_loss didn't improve as quickly. I adjusted a few times until I reached .01

learning_rate = 0.01
batch_size = 32
num_epochs = 30
steps_per_epoch = 200
validation_steps = 50
workers = 1

![alt_text][image1]

### Performance Results

![alt_text][image3]

![alt_text][image2]

The model could be improved with additional training data both following the hero in dense crowded areas, and also in larger patrol paths with the hero appearing less frequently. Tracking a dog or car could also be completed by the model but the training would need to include other animals/vehicles in an alternate environment and with different patrol patterns.