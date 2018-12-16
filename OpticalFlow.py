import os
import re
import random
import cv2 as cv
import numpy as np
import tensorflow as tf
import win_unicode_console

win_unicode_console.enable()

class OpticFlow:
    def __init__(self,Datastr):
        self.CLASS_NUM = 51
        self.tfrecordVideo()

        self.tfrecordVideo()

        # self.dataProcess(Datastr)

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
    def tfrecordVideo(self):
        
        # mFile = tf.data.TFRecordDataset(["train00.tfrecord", "train0A(1).tfrecord"])
        # iterTf = mFile.make_one_shot_iterator()
        # x = iterTf.get_next()
        # print(str(x['mean_rgb']))

        read_iterator = tf.python_io.tf_record_iterator(path= "train00.tfrecord")
        for record in read_iterator:
            example = tf.train.Example()
            example.ParseFromString(record)

            # print("Example:")
            with open("logs.csv","w") as f:
                f.write(str(example))

            videoLabel = example.features.feature['labels'].int64_list
            print(videoLabel)
            break

        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     # sess.run(iterTf.initializer())
        #     # sess.run(mFile)
        #     print(sess.run(x))
        #     # print(sess.run(x))
        return
    def dataProcess(self, dataStr):
        classDirs = os.listdir(dataStr)
        for mDir in classDirs:
            midDir = os.path.join(dataStr, mDir)
            mFiles = os.listdir(midDir)
            print(mDir)
            for mFile in mFiles:
                mFileStr = os.path.join(midDir, mFile)
                self.videoToOpticFlow(mFileStr)
        return
    def videoToOpticFlow(self,vdStr):
        cap = cv.VideoCapture(vdStr)
        frameCount = cap.get(7)
        ret, frame1 = cap.read()
        prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255

        frameStep = int(frameCount / 11)
        
        
        #Path where the optical flows will be saved
        # OutPath = "Optictest//"+ os.path.splitext(vdStr)[0]
        outPath = vdStr.split('\\')
        outPath ="Optictest\\" + outPath[1] + "\\" + os.path.splitext(outPath[2])[0] + "\\"
        if(os.path.exists(outPath) == False):
            os.makedirs(outPath)

        ret, frame2 = cap.read()
        iterCount = 1
        while(ret == True):
            if(iterCount % frameStep == 0):
                next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
                flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
                hsv[...,0] = ang*180/np.pi/2
                hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
                rgb = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
                imgGray = cv.cvtColor(rgb,cv.COLOR_BGR2GRAY)

                flowPath = outPath + "opticflow" + str(iterCount) + ".jpg"
                cv.imwrite(flowPath,imgGray)
                prvs = next

                if(int(iterCount / frameStep) == 5):
                    cv.imwrite(outPath + "frame.jpg",frame2)
            ret, frame2 = cap.read()
            iterCount += 1
            
        cap.release()
        cv.destroyAllWindows()
        return
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
    def dataProcess2(self, dataDir):
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
if(__name__ == "__main__"):
    opticalProcess = OpticFlow("hmdb51")
    # pass

