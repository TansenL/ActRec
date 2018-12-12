import os
import re
import random
import cv2 as cv
import numpy as np
import win_unicode_console

win_unicode_console.enable()

class OpticFlow:
    def __init__(self,Datastr):
        self.dataProcess(Datastr)
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

if(__name__ == "__main__"):
    opticalProcess = OpticFlow("hmdb51")

