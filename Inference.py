import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

def imageNet(images):

    keepPro = 0.1
    weightType = tf.float32
    images = tf.cast(images, tf.float32)
    #The first CONV layer 224*224*3
    convImageKernel1 = tf.Variable(tf.random_normal([7,7,1,96],stddev=5e-2),name="Conv_Image_1",dtype= weightType)
    convImageBaise1 = tf.Variable(tf.random_normal([96]),dtype= weightType, name= "ConvBase_Iamge_1")
    imgNet = tf.nn.conv2d(images, convImageKernel1,[1,2,2,1], padding= 'SAME')
    imgNet = tf.nn.bias_add(imgNet, convImageBaise1)
    imgNet = tf.nn.relu(imgNet)
    imgNet = tf.nn.max_pool(imgNet,ksize= [1,3,3,1], strides= [1,2,2,1], padding = "SAME")
    imgNet = tf.nn.lrn(imgNet)

    #The Second CONV layer  56*56*96
    convImageKernel2 = tf.Variable(tf.random_normal([5,5,96,258], stddev= 5e-2), dtype= weightType, name= "convImageKernel2")
    convImageBaise2 = tf.Variable(tf.random_normal([258]),dtype= weightType, name= "convImageBaise2")
    imgNet = tf.nn.conv2d(imgNet, convImageKernel2, strides= [1,2,2,1], padding= 'SAME')
    imgNet = tf.nn.bias_add(imgNet, convImageBaise2)
    imgNet = tf.nn.relu(imgNet)
    imgNet = tf.nn.max_pool(imgNet, ksize= [1,3,3,1], strides= [1,2,2,1], padding= 'SAME')
    imgNet = tf.nn.lrn(imgNet)

    #The Third CONV layer 14*14*258
    convImageKernel3 = tf.Variable(tf.random_normal([3,3,258,512], stddev= 5e-2), dtype= weightType, name= "convImageKernel3")
    convImageBaise3 = tf.Variable(tf.random_normal([512]),dtype= weightType, name= "convImageBaise3")
    imgNet = tf.nn.conv2d(imgNet, convImageKernel3, strides= [1,1,1,1], padding= 'SAME')
    imgNet = tf.nn.bias_add(imgNet, convImageBaise3)
    imgNet = tf.nn.relu(imgNet)
    
    #The Fourth CONV layer 14*14*512
    convImageKernel4 = tf.Variable(tf.random_normal([3,3,512,512], stddev= 5e-2), dtype= weightType, name= "convImageKernel4")
    convImageBaise4 = tf.Variable(tf.random_normal([512]),dtype= weightType, name= "convImageBaise4")
    imgNet = tf.nn.conv2d(imgNet, convImageKernel4, strides= [1,1,1,1], padding= 'SAME')
    imgNet = tf.nn.bias_add(imgNet, convImageBaise4)
    imgNet = tf.nn.relu(imgNet)

    #The Fifth CONV layer 14*14*512
    convImageKernel5 = tf.Variable(tf.random_normal([3,3,512,512], stddev= 5e-2), dtype= weightType, name= "convImageKernel5")
    convImageBaise5 = tf.Variable(tf.random_normal([512]),dtype= weightType, name= "convImageBaise5")
    imgNet = tf.nn.conv2d(imgNet, convImageKernel5, strides= [1,1,1,1], padding= 'SAME')
    imgNet = tf.nn.bias_add(imgNet, convImageBaise5)
    imgNet = tf.nn.relu(imgNet)
    imgNet = tf.nn.max_pool(imgNet, ksize= [1,3,3,1], strides = [1,2,2,1], padding = 'SAME')

    #Flatten the tensor
    imgNet = tf.reshape(imgNet,[-1, 7*7*512])
    
    #The Full-Connect layer 1*25088
    fullConnectKernel1 = tf.Variable(tf.random_normal([7*7*512, 4096], stddev= 5e-2), dtype= weightType, name= "fullConnectKernel1")
    fullConnectBaise1 = tf.Variable(tf.random_normal([4096],stddev= 5e-2), dtype= weightType, name= "fullConnectBaise1")
    imgNet = tf.nn.xw_plus_b(imgNet,fullConnectKernel1, fullConnectBaise1)
    imgNet = tf.nn.dropout(imgNet,keepPro)

    #The Full-Connect layer 1*4096
    fullConnectKernel2 = tf.Variable(tf.random_normal([4096, 2048], stddev= 5e-2), dtype= weightType, name= "fullConnectKernel2")
    fullConnectBaise2 = tf.Variable(tf.random_normal([2048],stddev= 5e-2), dtype= weightType, name= "fullConnectBaise2")
    imgNet = tf.nn.xw_plus_b(imgNet,fullConnectKernel2, fullConnectBaise2)
    imgNet = tf.nn.relu(imgNet)
    imgNet = tf.nn.dropout(imgNet,keepPro)

    #The Final layer 1*2048
    # fullConnectKernel3 = tf.Variable(tf.random_normal([2048, 10], stddev= 5e-2), dtype= weightType, name= "fullConnectKernel3")
    # fullConnectBaise3 = tf.Variable(tf.random_normal([10],stddev= 5e-2), dtype= weightType, name= "fullConnectBaise3")
    # imgNet = tf.nn.xw_plus_b(imgNet,fullConnectKernel3, fullConnectBaise3)
    # imgNet = tf.nn.relu(imgNet)
    # imgNet = tf.nn.dropout(imgNet,keepPro)

    #The Softmax layer
    imgNet = tf.nn.softmax(imgNet)

    #Out 1*10
    return imgNet
def opflowNet(opflows):

    keepPro = 0.1
    weightType = tf.float32
    opflows = tf.cast(opflows, tf.float32)
    #The first CONV layer 224*224*3
    convImageKernel1 = tf.Variable(tf.random_normal([7,7,11,96],stddev=5e-2),name="Conv_Image_1",dtype= weightType)
    convImageBaise1 = tf.Variable(tf.random_normal([96]),dtype= weightType, name= "ConvBase_Iamge_1")
    imgNet = tf.nn.conv2d(opflows, convImageKernel1,[1,2,2,1], padding= 'SAME')
    imgNet = tf.nn.bias_add(imgNet, convImageBaise1)
    imgNet = tf.nn.relu(imgNet)
    imgNet = tf.nn.max_pool(imgNet,ksize= [1,3,3,1], strides= [1,2,2,1], padding = "SAME")
    imgNet = tf.nn.lrn(imgNet)

    #The Second CONV layer  56*56*96
    convImageKernel2 = tf.Variable(tf.random_normal([5,5,96,258], stddev= 5e-2), dtype= weightType, name= "convImageKernel2")
    convImageBaise2 = tf.Variable(tf.random_normal([258]),dtype= weightType, name= "convImageBaise2")
    imgNet = tf.nn.conv2d(imgNet, convImageKernel2, strides= [1,2,2,1], padding= 'SAME')
    imgNet = tf.nn.bias_add(imgNet, convImageBaise2)
    imgNet = tf.nn.relu(imgNet)
    imgNet = tf.nn.max_pool(imgNet, ksize= [1,3,3,1], strides= [1,2,2,1], padding= 'SAME')

    #The Third CONV layer 14*14*258
    convImageKernel3 = tf.Variable(tf.random_normal([3,3,258,512], stddev= 5e-2), dtype= weightType, name= "convImageKernel3")
    convImageBaise3 = tf.Variable(tf.random_normal([512]),dtype= weightType, name= "convImageBaise3")
    imgNet = tf.nn.conv2d(imgNet, convImageKernel3, strides= [1,1,1,1], padding= 'SAME')
    imgNet = tf.nn.bias_add(imgNet, convImageBaise3)
    imgNet = tf.nn.relu(imgNet)
    
    #The Fourth CONV layer 14*14*512
    convImageKernel4 = tf.Variable(tf.random_normal([3,3,512,512], stddev= 5e-2), dtype= weightType, name= "convImageKernel4")
    convImageBaise4 = tf.Variable(tf.random_normal([512]),dtype= weightType, name= "convImageBaise4")
    imgNet = tf.nn.conv2d(imgNet, convImageKernel4, strides= [1,1,1,1], padding= 'SAME')
    imgNet = tf.nn.bias_add(imgNet, convImageBaise4)
    imgNet = tf.nn.relu(imgNet)

    #The Fifth CONV layer 14*14*512
    convImageKernel5 = tf.Variable(tf.random_normal([3,3,512,512], stddev= 5e-2), dtype= weightType, name= "convImageKernel5")
    convImageBaise5 = tf.Variable(tf.random_normal([512]),dtype= weightType, name= "convImageBaise5")
    imgNet = tf.nn.conv2d(imgNet, convImageKernel5, strides= [1,1,1,1], padding= 'SAME')
    imgNet = tf.nn.bias_add(imgNet, convImageBaise5)
    imgNet = tf.nn.relu(imgNet)
    imgNet = tf.nn.max_pool(imgNet, ksize= [1,3,3,1], strides = [1,2,2,1], padding = 'SAME')

    #Flatten the tensor
    imgNet = tf.reshape(imgNet,[-1, 7*7*512])
    
    #The Full-Connect layer 1*25088
    fullConnectKernel1 = tf.Variable(tf.random_normal([7*7*512, 4096], stddev= 5e-2), dtype= weightType, name= "fullConnectKernel1")
    fullConnectBaise1 = tf.Variable(tf.random_normal([4096],stddev= 5e-2), dtype= weightType, name= "fullConnectBaise1")
    imgNet = tf.nn.xw_plus_b(imgNet,fullConnectKernel1, fullConnectBaise1)
    imgNet = tf.nn.dropout(imgNet,keepPro)

    #The Full-Connect layer 1*4096
    fullConnectKernel2 = tf.Variable(tf.random_normal([4096, 2048], stddev= 5e-2), dtype= weightType, name= "fullConnectKernel2")
    fullConnectBaise2 = tf.Variable(tf.random_normal([2048],stddev= 5e-2), dtype= weightType, name= "fullConnectBaise2")
    imgNet = tf.nn.xw_plus_b(imgNet,fullConnectKernel2, fullConnectBaise2)
    imgNet = tf.nn.relu(imgNet)
    imgNet = tf.nn.dropout(imgNet,keepPro)

    #The Final layer 1*2048
    # fullConnectKernel3 = tf.Variable(tf.random_normal([2048, 10], stddev= 5e-2), dtype= weightType, name= "fullConnectKernel3")
    # fullConnectBaise3 = tf.Variable(tf.random_normal([10],stddev= 5e-2), dtype= weightType, name= "fullConnectBaise3")
    # imgNet = tf.nn.xw_plus_b(imgNet,fullConnectKernel3, fullConnectBaise3)
    # imgNet = tf.nn.relu(imgNet)
    # imgNet = tf.nn.dropout(imgNet,keepPro)

    #The Softmax layer
    imgNet = tf.nn.softmax(imgNet)

    return imgNet
def inference(images):

    imgNet = imageNet(images)
    # opflowsNet = opflowNet(opflows)

    #Fusion layer
    fusionKernel1 = tf.Variable(tf.random_normal([2048, 10], stddev= 5e-2), dtype= tf.float32, name= "fusionKernel1")
    fusionBaise1 = tf.Variable(tf.random_normal([10],stddev= 5e-2), dtype= tf.float32, name= "fusionBaise1")
    inferNet = tf.nn.xw_plus_b(imgNet,fusionKernel1, fusionBaise1)

    return inferNet

def trainInfer(images):

    classSize = 10
    learnRate = 0.01

    imageIn = tf.placeholder(tf.float32, shape= [None, 224, 224, 1])
    imageOut = tf.placeholder(tf.float32, shape= [None, classSize])

    inferNet = inference(imageIn)


    varList = [v for v in tf.trainable_variables()]
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= inferNet, labels= imageOut))
    # gradients = tf.gradients(loss, varList)
    train_op = tf.train.GradientDescentOptimizer(learnRate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        

def loadMnist(dataStr):

    mnist = input_data.read_data_sets(dataStr, one_hot= True)
    trainImg = np.vstack([img.reshape(-1,28,28,1) for img in mnist.train.images])
    trainLabel = mnist.train.labels

    # testImg = np.vstack([img.reshape(-1,28,28,1) for img in mnist.test.images])
    # testLabel = mnist.test.labels
    
    trainImg2 = []
    for i in range(trainImg.shape[0]):
        rainImg1 = cv2.resize(trainImg[0], dsize= (224,224)).tolist()
        trainImg2.append(rainImg1)

    print(trainImg2[0])

    

    
    return trainImg, trainLabel
        



if(__name__ == "__main__"):
    # a = np.ones([1,224,224,1])
    # flows = np.ones([1,224,224,11])
    # trainInfer(a)
    # loadMnist("minist/")
    pass