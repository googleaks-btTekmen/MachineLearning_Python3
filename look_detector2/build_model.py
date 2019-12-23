from keras.models import Sequential #to initialize model
from keras.layers.convolutional import Conv2D #performs convolution (matrix-matrix multiplication)
from keras.layers.convolutional import MaxPooling2D #reduces input size for each layer
from keras.layers.core import Activation #ReLu activation function
from keras.layers.core import Flatten #
from keras.layers.core import Dense #to define fully connected layers
from keras import backend as K

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        #initialize model
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        #first set of CONV => RELU => POOL layers
        model.add(Conv2D(20, (5,5), padding="same",
                         input_shape=inputShape)) #???what does "same" mean for padding
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2))) #2x2lik framede yatay ve dikey 2'ser adimlar atarak poolla

        #second layer
        model.add(Conv2D(50,(5,5), padding="same"))

        #third layer, first fully connected layer
        model.add(Flatten())
        model.add(Dense(500)) #how we decided to number 500?
        model.add(Activation("relu"))

        #what is difference between relu and softmax?
        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model






