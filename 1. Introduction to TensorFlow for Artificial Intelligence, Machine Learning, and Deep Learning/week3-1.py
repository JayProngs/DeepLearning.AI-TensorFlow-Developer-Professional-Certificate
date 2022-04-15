#!/usr/bin/env python
# coding: utf-8

# # Week 3: Improve MNIST with Convolutions
# 
# In the videos you looked at how you would improve Fashion MNIST using Convolutions. For this exercise see if you can improve MNIST to 99.5% accuracy or more by adding only a single convolutional layer and a single MaxPooling 2D layer to the model from the  assignment of the previous week. 
# 
# You should stop training once the accuracy goes above this amount. It should happen in less than 10 epochs, so it's ok to hard code the number of epochs for training, but your training must end once it hits the above metric. If it doesn't, then you'll need to redesign your callback.
# 
# When 99.5% accuracy has been hit, you should print out the string "Reached 99.5% accuracy so cancelling training!"
# 

# In[8]:


import os
import numpy as np
import tensorflow as tf
from tensorflow import keras


# Begin by loading the data. A couple of things to notice:
# 
# - The file `mnist.npz` is already included in the current workspace under the `data` directory. By default the `load_data` from Keras accepts a path relative to `~/.keras/datasets` but in this case it is stored somewhere else, as a result of this, you need to specify the full path.
# 
# - `load_data` returns the train and test sets in the form of the tuples `(x_train, y_train), (x_test, y_test)` but in this exercise you will be needing only the train set so you can ignore the second tuple.

# In[9]:


# Load the data

# Get current working directory
current_dir = os.getcwd() 

# Append data/mnist.npz to the previous path to get the full path
data_path = os.path.join(current_dir, "data/mnist.npz") 

# Get only training set
(training_images, training_labels), _ = tf.keras.datasets.mnist.load_data(path=data_path) 


# One important step when dealing with image data is to preprocess the data. During the preprocess step you can apply transformations to the dataset that will be fed into your convolutional neural network.
# 
# Here you will apply two transformations to the data:
# - Reshape the data so that it has an extra dimension. The reason for this 
# is that commonly you will use 3-dimensional arrays (without counting the batch dimension) to represent image data. The third dimension represents the color using RGB values. This data might be in black and white format so the third dimension doesn't really add any additional information for the classification process but it is a good practice regardless.
# 
# 
# - Normalize the pixel values so that these are values between 0 and 1. You can achieve this by dividing every value in the array by the maximum.
# 
# Remember that these tensors are of type `numpy.ndarray` so you can use functions like [reshape](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html) or [divide](https://numpy.org/doc/stable/reference/generated/numpy.divide.html) to complete the `reshape_and_normalize` function below:

# In[10]:


# GRADED FUNCTION: reshape_and_normalize

def reshape_and_normalize(images):
    
    ### START CODE HERE

    # Reshape the images to add an extra dimension
    images = images.reshape(60000,28,28,1)
    
    # Normalize pixel values
    images = images / 255.0
    
    ### END CODE HERE

    return images


# Test your function with the next cell:

# In[11]:


# Reload the images in case you run this cell multiple times
(training_images, _), _ = tf.keras.datasets.mnist.load_data(path=data_path) 

# Apply your function
training_images = reshape_and_normalize(training_images)

print(f"Maximum pixel value after normalization: {np.max(training_images)}\n")
print(f"Shape of training set after reshaping: {training_images.shape}\n")
print(f"Shape of one image after reshaping: {training_images[0].shape}")


# **Expected Output:**
# ```
# Maximum pixel value after normalization: 1.0
# 
# Shape of training set after reshaping: (60000, 28, 28, 1)
# 
# Shape of one image after reshaping: (28, 28, 1)
# ```

# Now complete the callback that will ensure that training will stop after an accuracy of 99.5% is reached:

# In[12]:


# GRADED CLASS: myCallback
### START CODE HERE

# Remember to inherit from the correct class
# class myCallback(tf.keras.callbacks.Callback):
#     def on_epoch_end(self,epoch,logs={}):
#         if(logs.get('acc')>0.995):
#             print('cancelling training')
#             self.model.stop_training=True
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') is not None and logs.get('accuracy') > 0.995:
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True
### END CODE HERE



# Finally, complete the `convolutional_model` function below. This function should return your convolutional neural network:

# In[13]:


# GRADED FUNCTION: convolutional_model
def convolutional_model():
    ### START CODE HERE

    # Define the model, it should have 5 layers:
    # - A Conv2D layer with 32 filters, a kernel_size of 3x3, ReLU activation function
    #    and an input shape that matches that of every image in the training set
    # - A MaxPooling2D layer with a pool_size of 2x2
    # - A Flatten layer with no arguments
    # - A Dense layer with 128 units and ReLU activation function
    # - A Dense layer with 10 units and softmax activation function
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape = (28,28,1)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ]) 

    ### END CODE HERE

    # Compile the model
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy']) 
        
    return model


# In[14]:


# Save your untrained model
model = convolutional_model()

# Instantiate the callback class
callbacks = myCallback()

# Train your model (this can take up to 5 minutes)
history = model.fit(training_images, training_labels, epochs=10, callbacks=[callbacks])


# If you see the message that you defined in your callback printed out after less than 10 epochs it means your callback worked as expected. You can also double check by running the following cell:

# In[15]:


print(f"Your model was trained for {len(history.epoch)} epochs")


# **Congratulations on finishing this week's assignment!**
# 
# You have successfully implemented a CNN to assist you in the image classification task. Nice job!
# 
# **Keep it up!**

# In[ ]:




