<h1 align="center"> Cataract-Detection</h1>

# Introduction

A novel cause to detect Cataract of a person.
With the help of this module a person can detect even a slightly occurence
of cataract by which he/she can separate cataract images from the non-cataract
images using the machine learning techniques to separate those images.
This uses classification techniques to separate those images ie. It separates 
cataract eyes from the non-cataract ones using keras library.

Certain steps that are followed to perform this project

<h4>Step 1.</h4>

First step involves collection of datasets thanks to the google images.I had been able to
collect cataract images of eyes of peoples and then similar process is to collect non-cataract images
so as to separate those images.Which also involves creation of training,testing and validation sets
So as to keep our training and test sets separate to learn our model from these images

<h4>Step 2.</h4>

After collection of the datasets and then training our model on the above collected dataset with the
help of mathematical computation library which adds the Dense layers for separation of those images
by adding 3 dense layers with the 2 as input layers with the relu as the input layers and then sigmoid
for the output layer then by giving 100 batches and 100 epochs for training our models with the adam
optimizers for optimization of our images to learn from them. Then after training the model we will save our
model to make our model more robust for training purpose and then testing our model for finding the accuracy 
how our model is trained.

<h4>Step 3.</h4>

Then reloading our model again to find the accuracy on the real time images of the person which uses computer
vision image processing techniques for separating those images and telling whether our eyes has cataract or not.


