#init mnist data and this step is very import for us
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data/",one_hot=True)

# realize the model of softmax that match mnist, but why the model use this realization is a question for me.
import tensorflow as tf

x=tf.placeholder(tf.float32,[None,784])
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))

# this is the realization of softmax: y=softmax(Wx+b)
y=tf.nn.softmax(tf.matmul(x,W)+b)

# now we shold make the function to train our mnist

# first we should compute cross entropy: corss_entropy=-sigma(y_*log(y))
y_=tf.placeholder("float",[None,10])
cross_entropy=-tf.reduce_sum(y_*tf.log(y))

# second we will realize the train step and we will use the gradient descent optimizer
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# and we alse need initialize function to init our variable
init=tf.initialize_all_variables()

# use session to make our function run
sess=tf.Session()
# run init variable function
sess.run(init)

# now we will train
import time
start_time = time.time()
for i in range(1000):
 batch_xs,batch_ys=mnist.train.next_batch(100)
 # run the function of train and input the placeholder x and y_
 sess.run(train_step,feed_dict={x:batch_xs,y_batch_ys})
 print("now we train mnist %d\%  time %dms"%((i/1000)*100)%(time.time()-start_time),)

# finally we need evaluate our mnist model
cross_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(cross_prediction,"float"))

print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))

# we alse need much work to do such as we need perfect this python to realize this: give the py a input, a picture of number, and the py give us a output, the number is what

print("to be continue...")
