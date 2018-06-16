##########  simple Multi-layer percepetron ( simple type of Neural Network)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()


# streaming training data
x  = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])  # actual values


# Weight tensor
W = tf.Variable(tf.zeros([784,10],tf.float32),name="weights")
# Bias tensor
b = tf.Variable(tf.zeros([10],tf.float32),name="biases")


# run the op initialize_all_variables using an interactive session
sess.run(tf.global_variables_initializer())

y = tf.nn.softmax(tf.matmul(x,W) + b) # predicted values

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#Load 50 training examples for each training iteration   
for i in range(500):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    #print("Weight matrix (learned parameters) \n",sess.run(tf.reduce_mean(W)))
    #print("\n\nBiases matrix (learned parameters) \n",sess.run(b))
    y_hat_softmax = tf.nn.softmax(tf.matmul(batch[0],W) + b)
    total_loss = tf.reduce_mean(-tf.reduce_sum(batch[1] * tf.log(y_hat_softmax), [1]))
    print("\nCross Entropy : ",sess.run(total_loss))

    
    
# evaluating the model

# HERE  y = tf.nn.softmax(tf.matmul(x,W) + b)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}) * 100
print("The final accuracy for the simple ANN model is: {} % ".format(acc) )



