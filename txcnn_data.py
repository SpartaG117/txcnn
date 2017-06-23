import pickle
import numpy as np
import cv2 as cv
import random
import tensorflow as tf
import time
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


    
    
def image_augment(data,label,num):
    sess = tf.InteractiveSession()
    data_aug=np.zeros((num*4,256,256,1))
    label_aug=np.zeros((num*4,2))
    data=data.reshape(num,256,256,1)
    data=tf.convert_to_tensor(data,dtype=tf.float32)
    label=tf.convert_to_tensor(label,dtype=tf.float32)
    
    data_ts=tf.Variable(tf.zeros([1,256,256,1]),dtype=tf.float32)
    label_ts=tf.Variable(tf.zeros([1,2]),dtype=tf.float32)
    
    tf.global_variables_initializer().run()
    print('data aumenting......')
    start=time.time()        
    for i in range(num):
        s=time.time()
        print('augumenting image:',i)
        img=data[i]
        img1=tf.reshape(img,[1,256,256,1])
        l=tf.reshape(label[i],[1,2])
        data_ts=tf.concat([data_ts,img1],0)
        label_ts=tf.concat([label_ts,l],0)

        a=time.time()
        img=tf.image.transpose_image(data[i])    #对角线翻转 
        img1=tf.reshape(img,[1,256,256,1])
        data_ts=tf.concat([data_ts,img1],0)
        label_ts=tf.concat([label_ts,l],0)

        img=tf.image.random_flip_left_right(data[i])   #随机左右翻转 
        img1=tf.reshape(img,[1,256,256,1])
        data_ts=tf.concat([data_ts,img1],0)
        label_ts=tf.concat([label_ts,l],0)

        img=tf.image.random_flip_up_down(data[i])      #随机上下翻转
        img1=tf.reshape(img,[1,256,256,1])
        data_ts=tf.concat([data_ts,img1],0)
        label_ts=tf.concat([label_ts,l],0)
        
        print('finished image:',i,'time:',time.time()-s)   
    
    data_np=data_ts[1:num*4+1]
    label_np=label_ts[1:num*4+1]
    data_np = data_np.eval()
    label_np = label_np.eval()
    print('data augmenting is finished','it take time:',time.time()-start)
    print(data_np.shape,label_np.shape)
    return data_np,label_np

def image_augment_serialize(training_data,training_label,num):
    data=np.zeros((num*4,256,256,1))
    label=np.zeros((num*4,2))
    for i in range(int(num/1000)):
        data[i*4000:(i+1)*4000],label[i*4000:(i+1)*4000] = image_augment(
                                            training_data[i*1000:(i+1)*1000],training_label[i*1000:(i+1)*1000],1000)
    
    data=data.reshape(num*4,256,256)
    for i in range(int(num*4/2000)):
        with open('./data/training_data_aug'+str(i)+'.pkl','wb') as f:
            print('save data serialization num',i)
            pickle.dump(data[i*2000:(i+1)*2000],f)
    with open('./data/training_label_aug.pkl','wb') as f:
        print('save label serialization')
        pickle.dump(label,f)
        
'''   
train,train_label,valid,valid_label=sample(training_data,training_label,2800,2500/3000,2500/300)
train=train.reshape(2800,256*256)
valid=valid.reshape(200,256*256)
'''
image_augment_serialize(training_data,training_label,3000)
