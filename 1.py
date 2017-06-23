import pickle
import numpy as np
import cv2 as cv
import random
import tensorflow as tf

#x1 = np.arange(6)
#print(np.sum(x1))

y1=np.array([[0,1],[2,1],[1,1]])
y2=np.array([[1,1],[1,2],[1,0]])
y = tf.placeholder(tf.float32,[3,2])
y_  = tf.placeholder(tf.float32,[3,2])
out = y*y_

sess = tf.InteractiveSession()


print(y.eval(feed_dict={y:y1,y_:y2}),'\n\n')
print(y_.eval(feed_dict={y:y1,y_:y2}),'\n\n')
print(out.eval(feed_dict={y:y1,y_:y2}))














# ##########################cnn1
'''
# #########cov1

weight1 = variable_weight([11,11,1,32])
kernel1 = tf.nn.conv2d(x_image, weight1, [1,4,4,1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[32]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

# #########cov2
weight2 = variable_weight([5,5,32,64])
kernel2 = tf.nn.conv2d(pool1, weight2,[1,4,4,1],padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
pool2 = tf.nn.max_pool(conv2,ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

# #########cov3

weight3 = variable_weight([3,3,64,128])
kernel3 = tf.nn.conv2d(pool2, weight3,[1,4,4,1],padding='SAME')
bias3 = tf.Variable(tf.constant(0.1, shape=[128]))
conv3= tf.nn.relu(tf.nn.bias_add(kernel3,bias3))
pool3 = tf.nn.max_pool(conv3,ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

# #########fc1
reshape_pool3=tf.reshape(pool3,[batch_size,-1])
dim = reshape_pool3.get_shape()[1].value
weight4 = variable_weight([dim,128])
bias4 = tf.Variable(tf.constant(0.1,shape=[128]))
fc1 = tf.nn.relu(tf.matmul(reshape_pool3,weight4) + bias4)

# #########fc2
weight5 = variable_weight([128,64])
bias5 = tf.Variable(tf.constant(0.1,shape=[64]))
fc2 = tf.nn.relu(tf.matmul(fc1,weight5) + bias5)

# #########softmax
weight6 = variable_weight([64,2])
bias6 = tf.Variable(tf.constant(0.1,shape=[2]))
output = tf.nn.log_softmax(tf.matmul(fc2,weight6) + bias6)


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*output,reduction_indices=[1]))
trian_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)

#valid test
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(output,1))
accuracy_count = tf.reduce_sum(tf.cast(correct_prediction,tf.float32))

# ##########test
test = tf.argmax(output,1)
'''

'''
# ##########################cnn2
# ##################相对cnn1，增加卷积核数量，减少全连接层节点数量

batch_size = 100
#num_trian_batch = training_data.shape[0]/batch_size
#num_trian_batch = train.shape[0]/batch_size
#num_valid_batch = valid.shape[0]/batch_size

x = tf.placeholder(tf.float32,[batch_size,256*256])    
y = tf.placeholder(tf.float32,[batch_size,2])
        
x_image = tf.reshape(x,[-1,256,256,1])
sess = tf.InteractiveSession()
        
# #########cov1
weight1 = variable_weight([11,11,1,64])
kernel1 = tf.nn.conv2d(x_image, weight1, [1,4,4,1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

# #########cov2
weight2 = variable_weight([5,5,64,128])
kernel2 = tf.nn.conv2d(pool1, weight2,[1,4,4,1],padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[128]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
pool2 = tf.nn.max_pool(conv2,ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

# #########cov3

weight3 = variable_weight([3,3,128,192])
kernel3 = tf.nn.conv2d(pool2, weight3,[1,4,4,1],padding='SAME')
bias3 = tf.Variable(tf.constant(0.1, shape=[192]))
conv3= tf.nn.relu(tf.nn.bias_add(kernel3,bias3))
pool3 = tf.nn.max_pool(conv3,ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

# #########fc1
reshape_pool3=tf.reshape(pool3,[batch_size,-1])
dim = reshape_pool3.get_shape()[1].value
weight4 = variable_weight([dim,256])
bias4 = tf.Variable(tf.constant(0.1,shape=[256]))
fc1 = tf.nn.relu(tf.matmul(reshape_pool3,weight4) + bias4)

# #########fc2
weight5 = variable_weight([256,128])
bias5 = tf.Variable(tf.constant(0.1,shape=[128]))
fc2 = tf.nn.relu(tf.matmul(fc1,weight5) + bias5)

# #########softmax
weight6 = variable_weight([128,2])
bias6 = tf.Variable(tf.constant(0.1,shape=[2]))
output = tf.nn.log_softmax(tf.matmul(fc2,weight6) + bias6)


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*output,reduction_indices=[1]))
trian_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)

#valid test
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(output,1))
accuracy_count = tf.reduce_sum(tf.cast(correct_prediction,tf.float32))

# ##########test
test = tf.argmax(output,1)



结果：

[ 0.725  0.785  0.2    0.22   0.2    0.75   0.685  0.765  0.645  0.225
  0.73   0.645  0.6    0.74   0.775  0.765  0.75   0.715  0.69   0.59 ]


accuracy_average:  0.61

'''





# ##########################cnn3
# ##################相对cnn2，增加dropout层
'''
batch_size = 100
#num_trian_batch = training_data.shape[0]/batch_size
#num_trian_batch = train.shape[0]/batch_size
#num_valid_batch = valid.shape[0]/batch_size

x = tf.placeholder(tf.float32,[batch_size,256*256])    
y = tf.placeholder(tf.float32,[batch_size,2])
keep_prob = tf.placeholder(tf.float32)
        
x_image = tf.reshape(x,[-1,256,256,1])
sess = tf.InteractiveSession()
        
# #########cov1
weight1 = variable_weight([11,11,1,64])
kernel1 = tf.nn.conv2d(x_image, weight1, [1,4,4,1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

# #########cov2
weight2 = variable_weight([5,5,64,128])
kernel2 = tf.nn.conv2d(pool1, weight2,[1,4,4,1],padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[128]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
pool2 = tf.nn.max_pool(conv2,ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

# #########cov3

weight3 = variable_weight([3,3,128,192])
kernel3 = tf.nn.conv2d(pool2, weight3,[1,4,4,1],padding='SAME')
bias3 = tf.Variable(tf.constant(0.1, shape=[192]))
conv3= tf.nn.relu(tf.nn.bias_add(kernel3,bias3))
pool3 = tf.nn.max_pool(conv3,ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

# #########fc1
reshape_pool3=tf.reshape(pool3,[batch_size,-1])
dim = reshape_pool3.get_shape()[1].value
weight4 = variable_weight([dim,256])
bias4 = tf.Variable(tf.constant(0.1,shape=[256]))
fc1 = tf.nn.relu(tf.matmul(reshape_pool3,weight4) + bias4)
fc1_drop = tf.nn.dropout(fc1,keep_prob)

# #########fc2
weight5 = variable_weight([256,128])
bias5 = tf.Variable(tf.constant(0.1,shape=[128]))
fc2 = tf.nn.relu(tf.matmul(fc1_drop,weight5) + bias5)

# #########softmax
weight6 = variable_weight([128,2])
bias6 = tf.Variable(tf.constant(0.1,shape=[2]))
output = tf.nn.log_softmax(tf.matmul(fc2,weight6) + bias6)


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*output,reduction_indices=[1]))
trian_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)

#valid test
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(output,1))
accuracy_count = tf.reduce_sum(tf.cast(correct_prediction,tf.float32))

# ##########test
test = tf.argmax(output,1)
'''




# ##########################cnn4
# ###########相比cnn1，增加dropout层,使用Adam优化器。


'''
# #########cov1

batch_size = 100
#num_trian_batch = training_data.shape[0]/batch_size
#num_trian_batch = train.shape[0]/batch_size
#num_valid_batch = valid.shape[0]/batch_size

x = tf.placeholder(tf.float32,[batch_size,256*256])    
y = tf.placeholder(tf.float32,[batch_size,2])
keep_prob = tf.placeholder(tf.float32)


weight1 = variable_weight([11,11,1,32])
kernel1 = tf.nn.conv2d(x_image, weight1, [1,4,4,1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[32]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

# #########cov2
weight2 = variable_weight([5,5,32,64])
kernel2 = tf.nn.conv2d(pool1, weight2,[1,4,4,1],padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
pool2 = tf.nn.max_pool(conv2,ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

# #########cov3

weight3 = variable_weight([3,3,64,128])
kernel3 = tf.nn.conv2d(pool2, weight3,[1,4,4,1],padding='SAME')
bias3 = tf.Variable(tf.constant(0.1, shape=[128]))
conv3= tf.nn.relu(tf.nn.bias_add(kernel3,bias3))
pool3 = tf.nn.max_pool(conv3,ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

# #########fc1
reshape_pool3=tf.reshape(pool3,[batch_size,-1])
dim = reshape_pool3.get_shape()[1].value
weight4 = variable_weight([dim,128])
bias4 = tf.Variable(tf.constant(0.1,shape=[128]))
fc1 = tf.nn.relu(tf.matmul(reshape_pool3,weight4) + bias4)
fc1_drop = tf.nn.dropout(fc1,keep_prob)

# #########fc2
weight5 = variable_weight([128,64])
bias5 = tf.Variable(tf.constant(0.1,shape=[64]))
fc2 = tf.nn.relu(tf.matmul(fc1_drop,weight5) + bias5)

# #########softmax
weight6 = variable_weight([64,2])
bias6 = tf.Variable(tf.constant(0.1,shape=[2]))
output = tf.nn.log_softmax(tf.matmul(fc2,weight6) + bias6)


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*output,reduction_indices=[1]))
trian_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#valid test
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(output,1))
accuracy_count = tf.reduce_sum(tf.cast(correct_prediction,tf.float32))

# ##########test
test = tf.argmax(output,1)


[ 0.715  0.715  0.725  0.72   0.68   0.58   0.69   0.74   0.705  0.695
  0.74   0.7    0.725  0.715  0.685  0.675  0.75   0.715  0.72   0.685]


accuracy_average:  0.70375


'''





