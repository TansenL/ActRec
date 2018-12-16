import os
import cv2
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as img

'''
ActRec: A class which can be used to Action Recognition, based on two-stream model

Methods:
fileToTensor: convert a image file to a tensor
dataInput: convert imagefiles to tensors
imageStream: A image stream network
opticalFlowStream: A optical flow network
infereNet: fusion of image stream network and optical flow network
actecTrain: train the two-stream model


'''
class ActRec:
    def __init__(self):
        self.BATCH_SIZE = 100
        self.IMAGE_SIZE = 256
        self.OPFLOW_DEEPTH = 10
        self.LEARN_RATE = 0.0001
        self.DROUP_OUT = 0.8
        self.TFVAR_TYPE = tf.float32
        self.EPOCH_NUM = 51             # EPOCH NUMBERS
        self.BATCH_NUM = 20             # BATCH PER EPOCH
        self.CLASS_NUM = 51
        self.ITER_NUM = 1000
        self.LRATE_DECAY = 0.1

        # self.train("DataSet")     # Train Model

        # self.actrecEval("DataSet")
    def imageStream(self,images,keepPro):

        '''Construct a images stream CNN network

        Args:
        images: a Placehoder

        Return:
        imgNet: a tensorflow CNN network

        '''

        weightType = self.TFVAR_TYPE
        images = tf.cast(images, weightType)
        #The first CONV layer 224*224*3
        with tf.variable_scope("ImageConv1") as scope:
            convImageKernel1 = tf.Variable(tf.random_normal([7,7,3,96],stddev=5e-2),name="Kernel",dtype= weightType)
            convImageBaise1 = tf.Variable(tf.random_normal([96]),dtype= weightType, name= "Baise")
            imgNet = tf.nn.conv2d(images, convImageKernel1,[1,2,2,1], padding= 'SAME')
            imgNet = tf.nn.bias_add(imgNet, convImageBaise1)
            imgNet = tf.nn.relu(imgNet, name= scope.name)
            imgNet = tf.nn.max_pool(imgNet,ksize= [1,3,3,1], strides= [1,2,2,1], padding = "SAME" )
            imgNet = tf.nn.lrn(imgNet, name= "LRN")

        #The Second CONV layer  56*56*96
        with tf.variable_scope("ImageConv2") as scope:
            convImageKernel2 = tf.Variable(tf.random_normal([5,5,96,258], stddev= 5e-2), dtype= weightType, name= "Kernel")
            convImageBaise2 = tf.Variable(tf.random_normal([258]),dtype= weightType, name= "Baise")
            imgNet = tf.nn.conv2d(imgNet, convImageKernel2, strides= [1,2,2,1], padding= 'SAME')
            imgNet = tf.nn.bias_add(imgNet, convImageBaise2)
            imgNet = tf.nn.relu(imgNet, name= scope.name)
            imgNet = tf.nn.max_pool(imgNet, ksize= [1,3,3,1], strides= [1,2,2,1], padding= 'SAME')
            imgNet = tf.nn.lrn(imgNet)

        #The Third CONV layer 14*14*258
        with tf.variable_scope("ImageConv3") as scope:
            convImageKernel3 = tf.Variable(tf.random_normal([3,3,258,512], stddev= 5e-2), dtype= weightType, name= "Kernel")
            convImageBaise3 = tf.Variable(tf.random_normal([512]),dtype= weightType, name= "Baise")
            imgNet = tf.nn.conv2d(imgNet, convImageKernel3, strides= [1,1,1,1], padding= 'SAME')
            imgNet = tf.nn.bias_add(imgNet, convImageBaise3)
            imgNet = tf.nn.relu(imgNet , name= scope.name)
        
        #The Fourth CONV layer 14*14*512
        with tf.variable_scope("ImageConv4") as scope:
            convImageKernel4 = tf.Variable(tf.random_normal([3,3,512,512], stddev= 5e-2), dtype= weightType, name= "Kernel")
            convImageBaise4 = tf.Variable(tf.random_normal([512]),dtype= weightType, name= "Baise")
            imgNet = tf.nn.conv2d(imgNet, convImageKernel4, strides= [1,1,1,1], padding= 'SAME')
            imgNet = tf.nn.bias_add(imgNet, convImageBaise4)
            imgNet = tf.nn.relu(imgNet, name= scope.name)

        #The Fifth CONV layer 14*14*512
        with tf.variable_scope("ImageConv5") as scope:
            convImageKernel5 = tf.Variable(tf.random_normal([3,3,512,512], stddev= 5e-2), dtype= weightType, name= "Kernel")
            convImageBaise5 = tf.Variable(tf.random_normal([512]),dtype= weightType, name= "Baise")
            imgNet = tf.nn.conv2d(imgNet, convImageKernel5, strides= [1,1,1,1], padding= 'SAME')
            imgNet = tf.nn.bias_add(imgNet, convImageBaise5)
            imgNet = tf.nn.relu(imgNet, name= scope.name)
            imgNet = tf.nn.max_pool(imgNet, ksize= [1,3,3,1], strides = [1,2,2,1], padding = 'SAME')

        #Flatten the tensor
        imgNet = tf.reshape(imgNet,[-1, 7*7*512])
        
        #The Full-Connect layer 1*25088
        with tf.variable_scope("ImageFC1") as scope:
            fullConnectKernel1 = tf.Variable(tf.random_normal([7*7*512, 4096], stddev= 5e-2), dtype= weightType, name= "Kernel")
            fullConnectBaise1 = tf.Variable(tf.random_normal([4096],stddev= 5e-2), dtype= weightType, name= "Baise")
            imgNet = tf.nn.xw_plus_b(imgNet,fullConnectKernel1, fullConnectBaise1)
            imgNet = tf.nn.relu(imgNet, name= scope.name)
            imgNet = tf.nn.dropout(imgNet,keepPro)

        #The Full-Connect layer 1*4096
        with tf.variable_scope("ImageFC2") as scope:
            fullConnectKernel2 = tf.Variable(tf.random_normal([4096, 2048], stddev= 5e-2), dtype= weightType, name= "Kernel")
            fullConnectBaise2 = tf.Variable(tf.random_normal([2048],stddev= 5e-2), dtype= weightType, name= "Baise")
            imgNet = tf.nn.xw_plus_b(imgNet,fullConnectKernel2, fullConnectBaise2)
            imgNet = tf.nn.relu(imgNet, name= scope.name)
            imgNet = tf.nn.dropout(imgNet,keepPro, name= "ImgNet")
        return imgNet
    def opticalFlowStream(self, opticalFlows, keepPro):
        ''' Construct a optical stream CNN network

        Args:
        opticalFlows: a Placeholder

        Returns:
        imgNet: a tensorflow optical stream CNN network
        '''

        weightType = self.TFVAR_TYPE
        opticalFlows = tf.cast(opticalFlows, weightType)
        #The first CONV layer 224*224*10
        with tf.variable_scope("OpticalConv1") as scope:
            convImageKernel1 = tf.Variable(tf.random_normal([7,7,10,96],stddev=5e-2),name="Kernel",dtype= weightType)
            convImageBaise1 = tf.Variable(tf.random_normal([96]),dtype= weightType, name= "Baise")
            imgNet = tf.nn.conv2d(opticalFlows, convImageKernel1,[1,2,2,1], padding= 'SAME')
            imgNet = tf.nn.bias_add(imgNet, convImageBaise1)
            imgNet = tf.nn.relu(imgNet, name= scope.name)
            imgNet = tf.nn.max_pool(imgNet,ksize= [1,3,3,1], strides= [1,2,2,1], padding = "SAME")
            imgNet = tf.nn.lrn(imgNet)

        #The Second CONV layer  56*56*96
        with tf.variable_scope("OpticalConv2") as scope:
            convImageKernel2 = tf.Variable(tf.random_normal([5,5,96,258], stddev= 5e-2), dtype= weightType, name= "Kernel")
            convImageBaise2 = tf.Variable(tf.random_normal([258]),dtype= weightType, name= "Baise")
            imgNet = tf.nn.conv2d(imgNet, convImageKernel2, strides= [1,2,2,1], padding= 'SAME')
            imgNet = tf.nn.bias_add(imgNet, convImageBaise2)
            imgNet = tf.nn.relu(imgNet, name= scope.name)
            imgNet = tf.nn.max_pool(imgNet, ksize= [1,3,3,1], strides= [1,2,2,1], padding= 'SAME')
            # imgNet = tf.nn.lrn(imgNet)

        #The Third CONV layer 14*14*258
        with tf.variable_scope("OpticalConv3") as scope:
            convImageKernel3 = tf.Variable(tf.random_normal([3,3,258,512], stddev= 5e-2), dtype= weightType, name= "Kernel")
            convImageBaise3 = tf.Variable(tf.random_normal([512]),dtype= weightType, name= "Baise")
            imgNet = tf.nn.conv2d(imgNet, convImageKernel3, strides= [1,1,1,1], padding= 'SAME')
            imgNet = tf.nn.bias_add(imgNet, convImageBaise3)
            imgNet = tf.nn.relu(imgNet, name= scope.name)
        
        #The Fourth CONV layer 14*14*512
        with tf.variable_scope("OpticalConv4") as scope:
            convImageKernel4 = tf.Variable(tf.random_normal([3,3,512,512], stddev= 5e-2), dtype= weightType, name= "Kernel")
            convImageBaise4 = tf.Variable(tf.random_normal([512]),dtype= weightType, name= "Baise")
            imgNet = tf.nn.conv2d(imgNet, convImageKernel4, strides= [1,1,1,1], padding= 'SAME')
            imgNet = tf.nn.bias_add(imgNet, convImageBaise4)
            imgNet = tf.nn.relu(imgNet, name= scope.name)

        #The Fifth CONV layer 14*14*512
        with tf.variable_scope("OpticalConv5") as scope:
            convImageKernel5 = tf.Variable(tf.random_normal([3,3,512,512], stddev= 5e-2), dtype= weightType, name= "Kernel")
            convImageBaise5 = tf.Variable(tf.random_normal([512]),dtype= weightType, name= "Baise")
            imgNet = tf.nn.conv2d(imgNet, convImageKernel5, strides= [1,1,1,1], padding= 'SAME')
            imgNet = tf.nn.bias_add(imgNet, convImageBaise5)
            imgNet = tf.nn.relu(imgNet, name= scope.name)
            imgNet = tf.nn.max_pool(imgNet, ksize= [1,3,3,1], strides = [1,2,2,1], padding = 'SAME')

        #Flatten the tensor
        imgNet = tf.reshape(imgNet,[-1, 7*7*512])
        
        #The Full-Connect layer 1*25088
        with tf.variable_scope("OpticalFC1") as scope:
            fullConnectKernel1 = tf.Variable(tf.random_normal([7*7*512, 4096], stddev= 5e-2), dtype= weightType, name= "Kernel")
            fullConnectBaise1 = tf.Variable(tf.random_normal([4096],stddev= 5e-2), dtype= weightType, name= "Baise")
            imgNet = tf.nn.xw_plus_b(imgNet,fullConnectKernel1, fullConnectBaise1)
            imgNet = tf.nn.relu(imgNet, name= scope.name)
            imgNet = tf.nn.dropout(imgNet,keepPro)

        #The Full-Connect layer 1*4096
        with tf.variable_scope("OpticalFC2") as scope:
            fullConnectKernel2 = tf.Variable(tf.random_normal([4096, 2048], stddev= 5e-2), dtype= weightType, name= "Kernel")
            fullConnectBaise2 = tf.Variable(tf.random_normal([2048],stddev= 5e-2), dtype= weightType, name= "Baise")
            imgNet = tf.nn.xw_plus_b(imgNet,fullConnectKernel2, fullConnectBaise2)
            imgNet = tf.nn.relu(imgNet, name= scope.name)
            imgNet = tf.nn.dropout(imgNet,keepPro)

        #The Softmax layer
        # imgNet = tf.nn.softmax(imgNet)

        return imgNet
    def infereNet(self, images, opflows, keepPro):
        '''Fusion image stream and optical stream into one network

        Args:
        images: image stream CNN network
        opflows: optical flow stream CNN network

        Returns:
        inferNet: Fusion Network without SoftMax

        '''

        imgNet = self.imageStream(images, keepPro)
        opflowsNet = self.opticalFlowStream(opflows, keepPro)
        fusiNet = tf.concat([imgNet, opflowsNet],1, name= "FusionConcat")

        #Fusion layer
        with tf.variable_scope("Fusion51") as scope:
            # fusiNet = tf.concat([imgNet, opflowsNet],1)
            fusionKernel = tf.Variable(tf.random_normal([4096,51], stddev= 5e-2), dtype= tf.float32, name= "Kernel")
            fusionBaise = tf.Variable(tf.random_normal([101],stddev= 5e-2), dtype= tf.float32, name= "Baise")
            inferNet51 = tf.nn.xw_plus_b(fusiNet,fusionKernel, fusionBaise, name= scope.name)
        with tf.variable_scope("Fusion101") as scope:
            fusionKernel101 = tf.Variable(tf.random_normal([4096,101], stddev= 5e-2), dtype= tf.float32, name= "Kernel")
            fusionBaise101 = tf.Variable(tf.random_normal([101], stddev= 5e-2), dtype= tf.float32, name= "Baise")
            inferNet101 = tf.nn.xw_plus_b(fusiNet,fusionKernel101,fusionBaise101, name= scope.name)
        return inferNet51, inferNet101
    def train(self,dataStr):
        ''' Train the defined model

        Args:
        dataStr: The Path of Dataset

        Returns:
        None

        '''
        
        #Placeholders
        tarinRate = tf.placeholder(tf.float32, name="LearnRate")
        trainDroup = tf.placeholder(tf.float32, name="DroupOut")
        trainImg = tf.placeholder(tf.float32,shape= [51,224,224,3], name="ImagesInput")
        trainOptical = tf.placeholder(tf.float32, shape= [51, 224, 224, 10], name= "OpticalInput")
        trainLabel = tf.placeholder(tf.float32, shape= [51, self.CLASS_NUM], name= "Labels")

        twoStreamNet51,twoStreamNet101 = self.infereNet(trainImg, trainOptical,trainDroup)

        #51 loss function and optimals
        loss51 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= twoStreamNet51, labels= trainLabel))
        # optimal = tf.train.GradientDescentOptimizer(tarinRate).minimize(loss)
        # optimal = tf.train.AdadeltaOptimizer(tarinRate).minimize(loss)
        optimal51 = tf.train.MomentumOptimizer(tarinRate, 0.9).minimize(loss51)
        acurrcy51 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(twoStreamNet51, 1), tf.argmax(trainLabel, 1)),tf.float32))

        #101 loss function and optimals
        loss101 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= twoStreamNet101, labels= trainLabel))
        optimal101 = tf.train.MomentumOptimizer(trainRate, 0.9).minimize(loss101)
        acurrcy101 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(twoStreamNet101, 1), tf.argmax(trainLabel, 1)), tf.float32))

        #Computing Part Begin
        try:
            with open("tmp\\logs.csv",'a') as f:
                f.write(str(time.localtime(time.time())) + "\tdroupout: 0.8\n" )
        except:
            pass

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            try:
                imageSet = os.listdir(dataStr + "\\images")
                images = dataStr + "\\images"
                opticals = dataStr + "\\opflows"
                imglabel = dataStr + "\\imglabel"
                iterCount = 0
                learRate = 0.01  
            except:
                pass

            for i in range(self.ITER_NUM):

                if(iterCount <=2000):
                    learRate = 0.0001
                elif(iterCount <= 10000):
                    learRate = 0.00005
                elif(iterCount <= 30000):
                    learRate = 0.00001
                elif(iterCount <= 60000):
                    learRate = 0.000005

                avgArr = 0
                avgCost = 0
                
                for batch in imageSet:

                    try:
                        fName = os.path.splitext(batch)[0]
                        fName = fName.split('_')[1]

                        imageIn = np.load(os.path.join(images, "images_" + str(fName) + ".npy"))
                        flowIn = np.load(os.path.join(opticals, "opflows_" + str(fName) + ".npy"))
                        imgLabel = np.load(os.path.join(imglabel, "imglabel_" + str(fName) + ".npy"))

                        _,cost,arr = sess.run([optimal51,loss51, acurrcy51], feed_dict= { trainImg : imageIn, trainOptical : flowIn, trainLabel : imgLabel, tarinRate : learRate, trainDroup : self.DROUP_OUT} )
                        
                        avgArr += arr
                        avgCost += cost
                    except:
                        pass
                    
                    iterCount += 1
                    
                    try:
                        if(iterCount <= 500):
                            print("Iter Count:" + str(iterCount) + "\tloss:" + str(cost) + "\taccuracy:" + str(arr))
                    except:
                        pass
                    # try:
                    #     if(iterCount % 1 == 0):
                    #         with open("tmp\\logs.csv",'a') as f:
                    #             f.write("Iter Count:" + str(iterCount) + "\tloss:" + str(cost) + "\taccuracy:" + str(arr) + "\n")
                    #         # print("Iter Count:" + str(iterCount) + "\tloss:" + str(cost) + "\taccuracy:" + str(arr))
                    # except:
                    #     pass
                try:
                    with open("tmp\\logs.csv",'a') as f:
                        f.write("Iter Count:" + str(i) + "\tloss:" + str(avgCost / len(imageSet)) + "\taccuracy:" + str(avgArr / len(imageSet)) + "\n")
                    if(iterCount % 1000 == 0):
                        saver.save(sess, "tmp/model.ckpt",global_step= iterCount)
                    if(float(avgArr / len(imageSet)) >= 0.98):
                        return
                except:
                    pass
        try:
            with open("tmp\\logs.csv",'a') as f:
                f.write(time.localtime(time.time()))
        except:
            pass

        return
    def actrecEval(self,dataStr):

        # #test model
        # correct_prediction=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
        # #calculate accuracy
        # accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        # print( "Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

        # trainDroup = tf.placeholder(tf.float32)
        # trainImg = tf.placeholder(tf.float32,shape= [51,224,224,3])
        # trainOptical = tf.placeholder(tf.float32, shape= [51, 224, 224, 10])
        # trainLabel = tf.placeholder(tf.float32, shape= [51, self.CLASS_NUM])
        # twoStreamNet = self.infereNet(trainImg, trainOptical,trainDroup)
        # twoStreamNet = tf.nn.softmax(twoStreamNet)
        # acurrcy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(twoStreamNet, 1), tf.argmax(trainLabel, 1)),tf.float32))


        
        # with tf.Session() as sess:
            
        #     sess.run(tf.global_variables_initializer())

        #     imageSet = os.listdir(dataStr + "\\images")
        #     images = dataStr + "\\images"
        #     opticals = dataStr + "\\opflows"
        #     imglabel = dataStr + "\\imglabel"
        #     iterCount = 0
        #     totalAccu = 0

        #     # imageIn = np.load(os.path.join(images, "images_1615.npy"))
        #     # flowIn = np.load(os.path.join(opticals, "opflows_1615.npy"))
        #     # imgLabel = np.load(os.path.join(imglabel, "imglabel_1615.npy"))
        #     saver = tf.train.import_meta_graph("tmp\\7000_without_droupOut\\model.ckpt.meta")
        #     saver.restore(sess, "tmp\\7000_without_droupOut\\model.ckpt")
        #     # print(sess.run(twoStreamNet, feed_dict= {trainImg : imageIn, trainOptical : flowIn, trainLabel : imgLabel})[3])
        #     # print(imgLabel[2])


        #     for batch in imageSet:

        #         fName = os.path.splitext(batch)[0]
        #         fName = fName.split('_')[1]

        #         imageIn = np.load(os.path.join(images, "images_" + str(fName) + ".npy"))
        #         flowIn = np.load(os.path.join(opticals, "opflows_" + str(fName) + ".npy"))
        #         imgLabel = np.load(os.path.join(imglabel, "imglabel_" + str(fName) + ".npy"))

        #         arr = sess.run(acurrcy, feed_dict= { trainImg : imageIn, trainOptical : flowIn, trainLabel : imgLabel, trainDroup : 1})

        #         print("Iter Count:\t" + str(iterCount) + "\t==================================\taccucy:\t" + str(arr * 100))

        #         totalAccu += arr
        #         iterCount += 1
        #     print("Total Accurecy:\t" + str(arr / 91 * 100 ))
            
        

        return
if(__name__ == "__main__"):
    a = ActRec()
