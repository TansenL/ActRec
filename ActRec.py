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

        self.actrecTrain("DataSet")     # Train Model

        # self.actrecEval("DataSet")

        # a = np.load("DataSet\\images_51.npy")
        # b = np.load("DataSet\\imglabel_51.npy")
        # c = np.load("DataSet\\opflows_51.npy")
        # print(a.shape)
        # print(b.shape)
        # print(c.shape)

        # self.dataProcess("OPtictest")
        # images = np.ones([2,224,224,3])
        # opflows = np.ones([2,224,224,11])
        # self.infereNet(images,opflows)
        # self.dataInput("Optictest")
    def fileToTensor(self, imageStr, IMAGE_SIZE ,flag = True ):
        '''From image file to a fixed-size tensor

        Args:
        imageStr: the location of image file
        IMAGE_SIZE: the size of processed image

        Return:
        imageTensor: 3D tensor of [IMAGE_SIZE, IMAGE_SIZE, CHANNEL_SIZE]
        '''
        # imageTensor = []
        # with tf.gfile.FastGFile(imageStr, 'rb') as f:
        #     imageRawData = f.read()
        # with tf.Session() as sess:
        #     imgData = tf.image.decode_jpeg(imageRawData)
        #     imgData = tf.image.convert_image_dtype(imgData,dtype = tf.float32)
        #     imgData = tf.image.resize_images(imgData, size = [256, 256],method= 0)
        #     imageTensor = imgData.eval()
        # print(imageTensor)

        imageTensor = cv2.imread(imageStr)
        imageTensor = imageTensor[:,:,[2,1,0]]
        imageTensor = cv2.resize(imageTensor, dsize= (IMAGE_SIZE,IMAGE_SIZE))
        if(flag == False):
            imageTensor = np.split(imageTensor,3,axis= 2)
            imageTensor = imageTensor[0]
        return imageTensor.tolist()
    def dataInput(self, dataDir):
        '''Construct train dataset for model

        Args:
        dataDir: Input dataset path

        Arrays:
        images: 4D tensor of [batch_size, IMAGE_WITH, IMAGE_HEIGHT, 3] (images dataset)
        opflows: 4D tensor of [batch_size, IMAGE_WITH, IMAGE_HEIGHT, 10] (optical flows dataset)
        imageLabels: 1D tensor of [batch_size] (image labels dataset)

        '''
        # images = []
        # opflows = []
        # imageLabels = []

        classDirs = os.listdir(dataDir)
        for mClass in classDirs:                                    #The Classes the image belong to
            
            print(mClass)

            images = []
            opflows = []
            imageLabels = []
            
            mClassStr = os.path.join(dataDir, mClass)
            mVideos = os.listdir(mClassStr)
            for mVideo in mVideos:                                  #The videoes in mClass
                mVideoStr = os.path.join(mClassStr, mVideo)
                mFrames = os.listdir(mVideoStr)

                mFlows = []
                for item in mFrames:                                #Frames in mVideo
                    if(item == mFrames[0]):
                        images.append(self.fileToTensor(os.path.join(mVideoStr,item),224))
                    elif(item == mFrames[1]):
                        mFlows = self.fileToTensor(os.path.join(mVideoStr,item), 224, False)
                    else:
                        mFlows = np.concatenate([mFlows,self.fileToTensor(os.path.join(mVideoStr,item),224,False)], axis= 2).tolist()
                        # mFlows.append(self.fileToTensor(os.path.join(mVideoStr,item),256))
                # mFlows = np.array(mFlows)
                opflows.append(mFlows)
                imageLabels.append(mClass)
                # print(opflows)
                # print(images)
                # print(imageLabels)
                # break  
            # break
            # print(mClass)
            
            ### Write data into .npy files
            # images = np.array(images)
            # opflows = np.array(opflows)
            # imageLabels = np.array(imageLabels)
            # print(opflows)
            np.save("DataSet\\images_"+mClass + ".npy",images)
            np.save("DataSet\\opflows_" + mClass + ".npy",opflows)
            np.save("DataSet\\imglabel_" + mClass + ".npy",imageLabels)
            
        return
    def dataProcess(self, dataDir):
        '''convert Image dataset to npy files

        Args:
        dataDir: the location where images are

        Return:
        None

        '''
        videoDict = dict()      #videos Names
        videoCount = 0          #Videos numbers
        labelDict = dict()      #Label To Onehot Vector

        classDirs = os.listdir(dataDir)
        for mClass in classDirs:
            mClassStr = os.path.join(dataDir, mClass)
            mVideos = os.listdir(mClassStr)
            for mVideo in mVideos:
                videoCount += 1
                if(videoDict.__contains__(mClass) == True):
                    if(videoDict[mClass].__contains__(mVideo) == False):
                        videoDict[mClass][mVideo] = True
                else:
                    midVideo = dict({mClass : mVideo})
                    videoDict[mClass] = midVideo
        
        #convert labels to onehot vectors
        onehotIter = 0
        for mKey in videoDict.keys():
            mLabel = np.zeros(self.CLASS_NUM)
            mLabel[onehotIter] = 1
            onehotIter += 1
            labelDict[mKey] = mLabel.tolist()
            with open("tmp\\oneHot.csv", 'a') as f:
                f.write(mKey + "\t" + str(mLabel.tolist()) + "\n")
        # print(labelDict)

        #process the images files
        while(videoCount > 0):

            print("iterCount:\t" + str(videoCount))
            midImg = []
            midFlow = []
            midLabel = []

            for item1 in videoDict.keys():
                # print(item1)
                for item2 in videoDict[item1]:
                    
                    if(videoDict[item1][item2] == True):

                        videoCount -= 1
                        mVideoStr = os.path.join(os.path.join(dataDir, item1),item2)
                        mFrames = os.listdir(mVideoStr)
                        mFlows = []
                        frameCount = 11

                        for mFrame in mFrames:
                            frameCount -= 1
                            if(frameCount >= 0):
                                if(mFrame == mFrames[0]):
                                    midImg.append(self.fileToTensor(os.path.join(mVideoStr,mFrame),224))
                                elif(mFrame == mFrames[1]):
                                    mFlows = self.fileToTensor(os.path.join(mVideoStr,mFrame), 224, False)
                                else:
                                    mFlows = np.concatenate([mFlows,self.fileToTensor(os.path.join(mVideoStr,mFrame),224,False)], axis= 2)
                                    # print(mFlows.shape)
                        
                        midFlow.append(mFlows)
                        videoDict[item1][item2] = False
                        break
                if(labelDict.__contains__(item1) == True):
                    midLabel.append(labelDict[item1])
                # break

            #Output images to npy files
            if(videoCount > 0):
                np.save("DataSet\\images_"+ str(videoCount) + ".npy",midImg)
                np.save("DataSet\\opflows_" + str(videoCount) + ".npy",midFlow)
                np.save("DataSet\\imglabel_" + str(videoCount) + ".npy",midLabel)
            # print(len(midLabel))
            # break
        return
    def imageStream(self,images,keepPro):

        '''Construct a images stream CNN network

        Args:
        images: a Placehoder

        Return:
        imgNet: a tensorflow CNN network

        '''

        weightType = tf.float32
        images = tf.cast(images, weightType)
        #The first CONV layer 224*224*3
        convImageKernel1 = tf.Variable(tf.random_normal([7,7,3,96],stddev=5e-2),name="Conv_Image_1",dtype= weightType)
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
        imgNet = tf.nn.relu(imgNet)
        imgNet = tf.nn.dropout(imgNet,keepPro)

        #The Full-Connect layer 1*4096
        fullConnectKernel2 = tf.Variable(tf.random_normal([4096, 2048], stddev= 5e-2), dtype= weightType, name= "fullConnectKernel2")
        fullConnectBaise2 = tf.Variable(tf.random_normal([2048],stddev= 5e-2), dtype= weightType, name= "fullConnectBaise2")
        imgNet = tf.nn.xw_plus_b(imgNet,fullConnectKernel2, fullConnectBaise2)
        imgNet = tf.nn.relu(imgNet)
        imgNet = tf.nn.dropout(imgNet,keepPro, name= "ImgNet")

        #The Softmax layer
        # imgNet = tf.nn.softmax(imgNet)
        #
        return imgNet
    def opticalFlowStream(self, opticalFlows, keepPro):
        ''' Construct a optical stream CNN network

        Args:
        opticalFlows: a Placeholder

        Returns:
        imgNet: a tensorflow optical stream CNN network
        '''

        weightType = tf.float32
        opticalFlows = tf.cast(opticalFlows, weightType)
        #The first CONV layer 224*224*10
        convImageKernel1 = tf.Variable(tf.random_normal([7,7,10,96],stddev=5e-2),name="Conv_Image_1",dtype= weightType)
        convImageBaise1 = tf.Variable(tf.random_normal([96]),dtype= weightType, name= "ConvBase_Iamge_1")
        imgNet = tf.nn.conv2d(opticalFlows, convImageKernel1,[1,2,2,1], padding= 'SAME')
        imgNet = tf.nn.bias_add(imgNet, convImageBaise1)
        imgNet = tf.nn.relu(imgNet)
        imgNet = tf.nn.max_pool(imgNet,ksize= [1,3,3,1], strides= [1,2,2,1], padding = "SAME")
        imgNet = tf.nn.lrn(imgNet, name= "ConvLayer1")

        #The Second CONV layer  56*56*96
        convImageKernel2 = tf.Variable(tf.random_normal([5,5,96,258], stddev= 5e-2), dtype= weightType, name= "convImageKernel2")
        convImageBaise2 = tf.Variable(tf.random_normal([258]),dtype= weightType, name= "convImageBaise2")
        imgNet = tf.nn.conv2d(imgNet, convImageKernel2, strides= [1,2,2,1], padding= 'SAME')
        imgNet = tf.nn.bias_add(imgNet, convImageBaise2)
        imgNet = tf.nn.relu(imgNet)
        imgNet = tf.nn.max_pool(imgNet, ksize= [1,3,3,1], strides= [1,2,2,1], padding= 'SAME', name= "ConvLayer2")
        # imgNet = tf.nn.lrn(imgNet)

        #The Third CONV layer 14*14*258
        convImageKernel3 = tf.Variable(tf.random_normal([3,3,258,512], stddev= 5e-2), dtype= weightType, name= "convImageKernel3")
        convImageBaise3 = tf.Variable(tf.random_normal([512]),dtype= weightType, name= "convImageBaise3")
        imgNet = tf.nn.conv2d(imgNet, convImageKernel3, strides= [1,1,1,1], padding= 'SAME')
        imgNet = tf.nn.bias_add(imgNet, convImageBaise3)
        imgNet = tf.nn.relu(imgNet, name= "ConvLayer3")
        
        #The Fourth CONV layer 14*14*512
        convImageKernel4 = tf.Variable(tf.random_normal([3,3,512,512], stddev= 5e-2), dtype= weightType, name= "convImageKernel4")
        convImageBaise4 = tf.Variable(tf.random_normal([512]),dtype= weightType, name= "convImageBaise4")
        imgNet = tf.nn.conv2d(imgNet, convImageKernel4, strides= [1,1,1,1], padding= 'SAME')
        imgNet = tf.nn.bias_add(imgNet, convImageBaise4)
        imgNet = tf.nn.relu(imgNet, name= "ConvLayer4")

        #The Fifth CONV layer 14*14*512
        convImageKernel5 = tf.Variable(tf.random_normal([3,3,512,512], stddev= 5e-2), dtype= weightType, name= "convImageKernel5")
        convImageBaise5 = tf.Variable(tf.random_normal([512]),dtype= weightType, name= "convImageBaise5")
        imgNet = tf.nn.conv2d(imgNet, convImageKernel5, strides= [1,1,1,1], padding= 'SAME')
        imgNet = tf.nn.bias_add(imgNet, convImageBaise5)
        imgNet = tf.nn.relu(imgNet)
        imgNet = tf.nn.max_pool(imgNet, ksize= [1,3,3,1], strides = [1,2,2,1], padding = 'SAME', name= "ConvLayer5")

        #Flatten the tensor
        imgNet = tf.reshape(imgNet,[-1, 7*7*512])
        
        #The Full-Connect layer 1*25088
        fullConnectKernel1 = tf.Variable(tf.random_normal([7*7*512, 4096], stddev= 5e-2), dtype= weightType, name= "fullConnectKernel1")
        fullConnectBaise1 = tf.Variable(tf.random_normal([4096],stddev= 5e-2), dtype= weightType, name= "fullConnectBaise1")
        imgNet = tf.nn.xw_plus_b(imgNet,fullConnectKernel1, fullConnectBaise1)
        imgNet = tf.nn.relu(imgNet, name= "FullConn1")
        imgNet = tf.nn.dropout(imgNet,keepPro)

        #The Full-Connect layer 1*4096
        fullConnectKernel2 = tf.Variable(tf.random_normal([4096, 2048], stddev= 5e-2), dtype= weightType, name= "fullConnectKernel2")
        fullConnectBaise2 = tf.Variable(tf.random_normal([2048],stddev= 5e-2), dtype= weightType, name= "fullConnectBaise2")
        imgNet = tf.nn.xw_plus_b(imgNet,fullConnectKernel2, fullConnectBaise2)
        imgNet = tf.nn.relu(imgNet, name= "FullConn2")
        imgNet = tf.nn.dropout(imgNet,keepPro, name= "OpflowNet")

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

        #Fusion layer
        fusiNet = tf.concat([imgNet, opflowsNet],1)
        fusionKernel = tf.Variable(tf.random_normal([4096,51], stddev= 5e-2), dtype= tf.float32, name= "fusionKernel")
        fusionBaise = tf.Variable(tf.random_normal([51],stddev= 5e-2), dtype= tf.float32, name= "fusionBaise")
        inferNet = tf.nn.xw_plus_b(fusiNet,fusionKernel, fusionBaise, name= "InferNet")
        # inferNet = tf.nn.relu(inferNet)
        return inferNet
    def actrecTrain(self,dataStr):
        ''' Train the defined model

        Args:
        dataStr: The Path of Dataset

        Returns:
        None

        '''
        
        #Placeholders
        tarinRate = tf.placeholder(tf.float32)
        trainDroup = tf.placeholder(tf.float32)
        trainImg = tf.placeholder(tf.float32,shape= [51,224,224,3])
        trainOptical = tf.placeholder(tf.float32, shape= [51, 224, 224, 10])
        trainLabel = tf.placeholder(tf.float32, shape= [51, self.CLASS_NUM])

        twoStreamNet = self.infereNet(trainImg, trainOptical,trainDroup)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= twoStreamNet, labels= trainLabel))
        # optimal = tf.train.GradientDescentOptimizer(tarinRate).minimize(loss)
        # optimal = tf.train.AdadeltaOptimizer(tarinRate).minimize(loss)
        optimal = tf.train.MomentumOptimizer(tarinRate, 0.9).minimize(loss)
        acurrcy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(twoStreamNet, 1), tf.argmax(trainLabel, 1)),tf.float32))

        saver = tf.train.Saver()
        try:
            with open("tmp\\logs.csv",'a') as f:
                f.write(str(time.localtime(time.time())) + "\tdroupout: 0.8\n" )
        except:
            pass
        
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

                        _,cost,arr = sess.run([optimal,loss, acurrcy], feed_dict= { trainImg : imageIn, trainOptical : flowIn, trainLabel : imgLabel, tarinRate : learRate, trainDroup : self.DROUP_OUT} )
                        
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
                    saver.save(sess, "tmp/model.ckpt")
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

        trainDroup = tf.placeholder(tf.float32)
        trainImg = tf.placeholder(tf.float32,shape= [51,224,224,3])
        trainOptical = tf.placeholder(tf.float32, shape= [51, 224, 224, 10])
        trainLabel = tf.placeholder(tf.float32, shape= [51, self.CLASS_NUM])
        twoStreamNet = self.infereNet(trainImg, trainOptical,trainDroup)
        twoStreamNet = tf.nn.softmax(twoStreamNet)
        acurrcy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(twoStreamNet, 1), tf.argmax(trainLabel, 1)),tf.float32))


        
        with tf.Session() as sess:
            
            sess.run(tf.global_variables_initializer())

            imageSet = os.listdir(dataStr + "\\images")
            images = dataStr + "\\images"
            opticals = dataStr + "\\opflows"
            imglabel = dataStr + "\\imglabel"
            iterCount = 0
            totalAccu = 0

            # imageIn = np.load(os.path.join(images, "images_1615.npy"))
            # flowIn = np.load(os.path.join(opticals, "opflows_1615.npy"))
            # imgLabel = np.load(os.path.join(imglabel, "imglabel_1615.npy"))
            saver = tf.train.import_meta_graph("tmp\\7000_without_droupOut\\model.ckpt.meta")
            saver.restore(sess, "tmp\\7000_without_droupOut\\model.ckpt")
            # print(sess.run(twoStreamNet, feed_dict= {trainImg : imageIn, trainOptical : flowIn, trainLabel : imgLabel})[3])
            # print(imgLabel[2])


            for batch in imageSet:

                fName = os.path.splitext(batch)[0]
                fName = fName.split('_')[1]

                imageIn = np.load(os.path.join(images, "images_" + str(fName) + ".npy"))
                flowIn = np.load(os.path.join(opticals, "opflows_" + str(fName) + ".npy"))
                imgLabel = np.load(os.path.join(imglabel, "imglabel_" + str(fName) + ".npy"))

                arr = sess.run(acurrcy, feed_dict= { trainImg : imageIn, trainOptical : flowIn, trainLabel : imgLabel, trainDroup : 1})

                print("Iter Count:\t" + str(iterCount) + "\t==================================\taccucy:\t" + str(arr * 100))

                totalAccu += arr
                iterCount += 1
            print("Total Accurecy:\t" + str(arr / 91 * 100 ))
            
        

        return
if(__name__ == "__main__"):
    a = ActRec()
