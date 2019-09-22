# From https://www.tensorflow.org/beta/

import tensorflow as tf
import time
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

t0 = time.time()

# Load and prepare the MNIST dataset.
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build the tf.keras.Sequential model by stacking layers, choose 
# loss function and optimizers for training.
model0 = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28)),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation = 'softmax')])
model0.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

# Train and evaluate the model.
model0.fit(x_train, y_train, epochs = 5)
result0 = model0.evaluate(x_test, y_test)
elapse0 = time.time() - t0
print(result0, elapse0)

# Use tf.data to batch and shuffle the dataset.
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices(
    (x_test, y_test)).batch(32)

# Build the tf.keras model using the Keras model subclassing API.
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation = 'relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation = 'relu')
        self.d2 = Dense(10, activation = 'softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

model1 = MyModel()

# Choose an optimizer and loss function for training.
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# Selecte metrics to measure the loss and the accuracy of the model. 
# These metrics accumulate the values over epochs and then print the overall result.
train_loss = tf.keras.metrics.Mean(name = 'train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalCrossentropy(name = 'train_accuracy')

test_loss = tf.keras.metrics.Mean(name = 'test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalCrossentropy(name = 'test_accuracy')

# Use GradientTape to train the model.
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model1(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model1.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model1.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

# Test the model.
@tf.function
def test_step(images, labels):
    predictions = model1(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

EPOCHS = 5

for epoch in range(EPOCHS):
    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))
    
    # Reset the metrics for the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

elapes1 = time.time() - elapse0 - t0
print(elapes1)
