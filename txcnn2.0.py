import pickle
import numpy as np
import cv2 as cv
import random
def unpickle(file):
    with open(file,'rb') as f:
        data=np.array([])
        data=pickle.load(f)
        return data


def read_serialize_training_data():
    img_data=np.zeros((3000,256,256))
    for i in range(3000):
        if i<=2599:
            print('read image '+str(i))
            img=cv.imread('./1/'+str(i)+'.png',cv.IMREAD_GRAYSCALE)
            img=cv.resize(img,(256,256))
            img_data[i]=img
            
        else:
            print('read image '+str(i))
            img=cv.imread('./0/'+str(i)+'.png',cv.IMREAD_GRAYSCALE)
            img=cv.resize(img,(256,256))
            img_data[i]=img
    with open('./data/training_data.pkl','wb') as f:
        print('save serialization')
        pickle.dump(img_data,f)

def read_serialize_test_data():
    img_data=np.zeros((5000,256,256))
    num=0
    j=0
    for i in range(1,100001):
        print('read test image '+str(i-1))
        img=cv.imread('./test/'+str(i-1)+'.png',cv.IMREAD_GRAYSCALE)
        img=cv.resize(img,(256,256))
        img_data[j]=img 
        if i%5000==0:
            with open('./data/testing_data'+str(num)+'.pkl','wb') as f:
                print('save serialization'+str(num))
                pickle.dump(img_data,f)
            num=num+1
            j=0
            continue
        j=j+1      


def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  index=(index_offset + labels_dense.ravel()).astype(int)
  labels_one_hot.flat[index] = 1
  return labels_one_hot

#read_serialize_training_data()    
#read_serialize_test_data()    

training_label=unpickle('./data/training_label.pkl')
#training_id=unpickle('./data/training_id.pkl')
training_label=dense_to_one_hot(training_label,2)
training_data=unpickle('./data/training_data.pkl')



def sample(data,label,num_samples,positive_ratio,positive_negative_sample_ratio=1.0):
    print('sampling....')
    num_sample_negatives=int(num_samples/(1+positive_negative_sample_ratio))
    num_sample_positives=int(num_sample_negatives * positive_negative_sample_ratio)
    #print(num_sample_negatives,num_sample_positives)
    index_sample_positives=random.sample(range(int(data.shape[0]*positive_ratio)),num_sample_positives)
    index_sample_negatives=random.sample(range(int(data.shape[0]*positive_ratio),data.shape[0]),num_sample_negatives)
    index_sample = index_sample_positives + index_sample_negatives
    index_sample.sort()
    
    training_sample=np.zeros((num_samples,data.shape[1],data.shape[2]))
    training_sample_label=np.zeros((num_samples,2))
    test_sample=np.zeros((data.shape[0]-num_samples,data.shape[1],data.shape[2]))
    test_sample_label=np.zeros((data.shape[0]-num_samples,2))
    
    j=0
    for i in index_sample:
        training_sample[j]=data[i]
        training_sample_label[j]=label[i]
        #print('finish',str(j))
        j=j+1
        
    #print(training_sample.shape)
    #print(training_sample_label)
    index_test=list(range(data.shape[0]-num_samples))
    j=0
    k=0
    for i in index_sample:
        while j<i :
            index_test[k]=j
            k=k+1
            j=j+1
        j=j+1
    while j<data.shape[0]:
        index_test[k]=j
        k=k+1
        j=j+1
        
    j=0
    for i in index_test:
        test_sample[j]=data[i]
        test_sample_label[j]=label[i]
        j=j+1
    print('finish sampling')
    return training_sample,training_sample_label,test_sample,test_sample_label
        
def next_batch(data,label,batch_size):
    num_data=data.shape[0]
    index1=random.sample(range(2500),int(batch_size/2))
    index2=random.sample(range(2500,2800),int(batch_size/2))
    batch_data=np.zeros((batch_size,256*256))
    batch_label=np.zeros((batch_size,2))
    j=0
    for i in index1:
        batch_data[j]=data[i]
        batch_label[j]=label[i]
        j=j+1
    for i in index2:
        batch_data[j]=data[i]
        batch_label[j]=label[i]
        j=j+1
    return batch_data,batch_label    





'''   
train,train_label,valid,valid_label=sample(training_data,training_label,2800,2500/3000,2500/300)
train=train.reshape(2800,256*256)
valid=valid.reshape(200,256*256)
'''


#training_data=training_data.reshape(3000,256*256)



import tensorflow as tf
import time


def variable_weight(shape,stddev):
        initial = tf.truncated_normal(shape, stddev)
        return tf.Variable(initial)


def training_func(data,label,valid,valid_label,turn):
    

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
    weight1 = variable_weight([11,11,1,32], stddev=0.05)
    kernel1 = tf.nn.conv2d(x_image, weight1, [1,4,4,1], padding='SAME')
    bias1 = tf.Variable(tf.constant(0.0, shape=[32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
    pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

    # #########cov2
    weight2 = variable_weight([5,5,32,64],stddev=0.05)
    kernel2 = tf.nn.conv2d(pool1, weight2,[1,4,4,1],padding='SAME')
    bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
    pool2 = tf.nn.max_pool(conv2,ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

    # #########cov3

    weight3 = variable_weight([3,3,64,128],stddev=0.05)
    kernel3 = tf.nn.conv2d(pool2, weight3,[1,4,4,1],padding='SAME')
    bias3 = tf.Variable(tf.constant(0.1, shape=[128]))
    conv3= tf.nn.relu(tf.nn.bias_add(kernel3,bias3))
    pool3 = tf.nn.max_pool(conv3,ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

    # #########fc1
    reshape_pool3=tf.reshape(pool3,[batch_size,-1])
    dim = reshape_pool3.get_shape()[1].value
    weight4 = variable_weight([dim,128], stddev=0.04)
    bias4 = tf.Variable(tf.constant(0.1,shape=[128]))
    fc1 = tf.nn.relu(tf.matmul(reshape_pool3,weight4) + bias4)
    fc1_drop = tf.nn.dropout(fc1,keep_prob)
    wl4 = 0.04
    weight4_loss = tf.multiply(tf.nn.l2_loss(weight4),wl4)

    # #########fc2
    weight5 = variable_weight([128,64], stddev=0.04)
    bias5 = tf.Variable(tf.constant(0.1,shape=[64]))
    fc2 = tf.nn.relu(tf.matmul(fc1_drop,weight5) + bias5)
    wl5 = 0.04
    weight5_loss = tf.multiply(tf.nn.l2_loss(weight5),wl5)
    
    # #########softmax
    weight6 = variable_weight([64,2], stddev=1/64.0)
    bias6 = tf.Variable(tf.constant(0.0,shape=[2]))
    output = tf.nn.log_softmax(tf.matmul(fc2,weight6) + bias6)
    
    
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*output,reduction_indices=[1])) + weight4_loss +weight5_loss
    
    trian_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    #valid test
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(output,1))
    accuracy_count = tf.reduce_sum(tf.cast(correct_prediction,tf.float32))

    # ##########test
    test = tf.argmax(output,1)




    tf.global_variables_initializer().run()


# #######################train


    print('begin trianing .......')

    start_time=time.time()
    max_steps=3000
    for i in range(max_steps):
        #b_data,b_label=next_batch(training_data,training_label,batch_size)
        b_data,b_label=next_batch(data,label,batch_size)
        trian_step.run(feed_dict={x:b_data,y:b_label,keep_prob:0.5})
        c=cross_entropy.eval(feed_dict={x:b_data,y:b_label,keep_prob:0.5})
        print('turn ',turn,'iter '+str(i+1),'cross_entropy:',c)
       
        #if (i>2000)and(c<=0.23):
        #    break

    print('training finished  takes time',time.time()-start_time)

    # ###########valid

    count=0
    for i in range(2):
        count = count + accuracy_count.eval(feed_dict={x: valid[i*batch_size:(i+1)*batch_size], 
                                              y: valid_label[i*batch_size:(i+1)*batch_size],
                                              keep_prob:1.0})
        
    print('valid accuracy:  ',count/valid.shape[0])
    
    return count/valid.shape[0]


train_num=50
accuracy_total=np.zeros((train_num))
for i in range(train_num):
                                    
    train,train_label,valid,valid_label=sample(training_data,training_label,2800,2500/3000,2500/300)
    train=train.reshape(2800,256*256)
    valid=valid.reshape(200,256*256)
    accuracy_total[i] = training_func(train,train_label,valid,valid_label,i+1)

accuracy_average=np.sum(accuracy_total)/train_num
print('\n\n\n\n\n\n\n')
print(accuracy_total,'\n\n')
print('accuracy_average: ',accuracy_average)
    
    
    
    
    
    

'''
a=output.eval(feed_dict={x:valid[0:100]})
b=output.eval(feed_dict={x:valid[100:200]})
print(np.e**a)
print(np.e**b)
print(valid_label)
'''



  
''' 
print('\n\n\n\n')
print('compute test image')
start=time.time()
test_label=np.ones((100000))
for i in range(20):
    test_data=np.zeros((5000,256,256))
    with open('./data/testing_data'+str(i)+'.pkl','rb') as f:
        print('compute',i)
        test_data=pickle.load(f)     
        test_data=test_data.reshape(5000,256*256)
        
    for k in range(50):
        index1=i*5000+k*100
        index2=i*5000+(k+1)*100
        test_label[index1:index2] = test.eval(feed_dict={x:test_data[k*100:(k+1)*100]})           
print('compute finish')
print('it takes time: ',str(time.time()-start))



print(test_label.shape)
with open('test.txt','w') as outfile:
    j=0
    num=0
    for i in test_label:
        if i == 0:
            outfile.write(str(j+1)+'\n')
            num=num+1
        j=j+1
    print('There are '+str(num)+' negatives')

'''



