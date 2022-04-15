# # Week 2: Implementing Callbacks in TensorFlow using the MNIST Dataset
# 
# In the course you learned how to do classification using Fashion MNIST, a data set containing items of clothing.
# There's another, similar dataset called MNIST which has items of handwriting -- the digits 0 through 9.
# 
# Write an MNIST classifier that trains to 99% accuracy or above, and does it without a fixed number of epochs --
# i.e. you should stop training once you reach that level of accuracy. In the lecture you saw how this was done for
# the loss but here you will be using accuracy instead.
# 
# Some notes: 1. Given the architecture of the net, it should succeed in less than 10 epochs. 2. When it reaches 99%
# or greater it should print out the string "Reached 99% accuracy so cancelling training!" and stop training. 3. If
# you add any additional variables, make sure you use the same names as the ones used in the class. This is important
# for the function signatures (the parameters and names) of the callbacks.

# In[32]:


import os
import tensorflow as tf
from tensorflow import keras

# Begin by loading the data. A couple of things to notice:
# 
# - The file `mnist.npz` is already included in the current workspace under the `data` directory. By default the
# `load_data` from Keras accepts a path relative to `~/.keras/datasets` but in this case it is stored somewhere else,
# as a result of this, you need to specify the full path.
# 
# - `load_data` returns the train and test sets in the form of the tuples `(x_train, y_train), (x_test, y_test)` but
# in this exercise you will be needing only the train set so you can ignore the second tuple.

# In[33]:


# Load the data

# Get current working directory
current_dir = os.getcwd()

# Append data/mnist.npz to the previous path to get the full path
data_path = os.path.join(current_dir, "data/mnist.npz")

# Discard test set
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data(path=data_path)

# Normalize pixel values
x_train = x_train / 255.0

# Now take a look at the shape of the training data:

# In[34]:


data_shape = x_train.shape

print(f"There are {data_shape[0]} examples with shape ({data_shape[1]}, {data_shape[2]})")


# Now it is time to create your own custom callback. For this complete the `myCallback` class and the `on_epoch_end`
# method in the cell below. If you need some guidance on how to proceed, check out this [link](
# https://www.tensorflow.org/guide/keras/custom_callback).

# In[35]:


# GRADED CLASS: myCallback
### START CODE HERE

# Remember to inherit from the correct class
class myCallback(tf.keras.callbacks.Callback):
    # Define the correct function signature for on_epoch_end
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') is not None and logs.get('accuracy') > 0.99:  # @KEEP
            print("\nReached 99% accuracy so cancelling training!")

            # Stop training once the above condition is met
            self.model.stop_training = True


### END CODE HERE


# Now that you have defined your callback it is time to complete the `train_mnist` function below:

# In[36]:


# GRADED FUNCTION: train_mnist
def train_mnist(x_train, y_train):
    ### START CODE HERE

    # Instantiate the callback class
    callbacks = myCallback()

    # Define the model, it should have 3 layers:
    # - A Flatten layer that receives inputs with the same shape as the images
    # - A Dense layer with 512 units and ReLU activation function
    # - A Dense layer with 10 units and softmax activation function
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(data_shape[1], data_shape[2])),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Fit the model for 10 epochs adding the callbacks
    # and save the training history
    history = model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])

    ### END CODE HERE

    return history


# Call the `train_mnist` passing in the appropiate parameters to get the training history:

# In[37]:


hist = train_mnist(x_train, y_train)

# If you see the message `Reached 99% accuracy so cancelling training!` printed out after less than 10 epochs it
# means your callback worked as expected.

# **Congratulations on finishing this week's assignment!**
# 
# You have successfully implemented a callback that gives you more control over the training loop for your model.
# Nice job!
# 
# **Keep it up!**
