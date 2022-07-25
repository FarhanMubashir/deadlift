from django.shortcuts import redirect, render

# Create your views here.
from tkinter import *
import cv2
import mediapipe as mp
import numpy as np
from numpy import savetxt
import time
import matplotlib.pyplot as plt
from django.shortcuts import redirect
from mpl_toolkits.mplot3d import Axes3D



def index(request):
    return render(request,'index.html')



def start(request):
    class poseDetector():

        def __init__(self, mode=False, modelComp = 1, smooth=True, enable_seg=False,
                        smooth_seg=False, detectionCon=0.5, trackCon=0.5):

            self.mode = mode
            self.modelComp = modelComp
            self.smooth = smooth
            self.enable_seg = enable_seg
            self.smooth_seg = smooth_seg
            self.detectionCon = detectionCon
            self.trackCon = trackCon

            self.mpDraw = mp.solutions.drawing_utils
            self.mpPose = mp.solutions.pose
            self.pose = self.mpPose.Pose(self.mode, self.modelComp, self.smooth, self.enable_seg, self.smooth_seg,
                                            self.detectionCon, self.trackCon)

        def findPose(self, img, draw=True):
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.pose.process(imgRGB)
            if self.results.pose_landmarks:
                if draw:
                    #self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                    #                            self.mpPose.POSE_CONNECTIONS)
                    IsPose = True
            else:
                    IsPose = False
            return img, IsPose

        def findPosition(self, img, draw=True):
            self.lmList = []
            self.world = self.results.pose_world_landmarks.landmark
            if self.results.pose_landmarks:
                for id, lm in enumerate(self.results.pose_landmarks.landmark):
                    h, w, c = img.shape
                    # print(id, lm)
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lmList.append([id, cx, cy])
                    #if draw:
                    #    cv2.circle(img, (cx, cy), 3, (255, 0, 0), cv2.FILLED)
            return self.lmList, self.world

        def findAngle(self, img, p1, p2, p3, draw=False):

            # Get the landmarks
            x1, y1 = self.lmList[p1][1:]
            x2, y2 = self.lmList[p2][1:]
            x3, y3 = self.lmList[p3][1:]

            joint1 = np.array([self.world[p1].x, self.world[p1].y, self.world[p1].z])
            joint2 = np.array([self.world[p2].x, self.world[p2].y, self.world[p2].z])
            joint3 = np.array([self.world[p3].x, self.world[p3].y, self.world[p3].z])

            # Calculate the Angle
            angle = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(joint1-joint2, joint3-joint2)),
                                                np.dot(joint1-joint2, joint3- joint2)))

            # print(angle)

            # Draw
            if draw:
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
                cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
                cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
                cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
                cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
                #cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                #            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

            return angle

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    calib = [0,0]
    disKnee = []
    calibNose = [320, 100]
    calibRHeel = [290, 420]
    calibLHeel = [340, 420]
    calbRes = 35
    status = 0
    countFrame = 0
    repDown = 0
    repUp = 1
    rep = 0

    arrow1 = cv2.imread("static/images/arrowLR_no_bg.png") # =>
    arrow2 = cv2.imread("static/images/arrowLR_no_bg.png") # <=
    arrow2 = cv2.flip(arrow2,1)
    arrow1 = cv2.resize(arrow1,(30,30))
    arrow2 = cv2.resize(arrow2,(30,30))

    fig = plt.figure()
    ax = Axes3D(fig,azim=-10)



    detector = poseDetector()


    while cap.isOpened():
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img, IsPose = detector.findPose(img, draw=True)
        frame_width = int(cap.get(3)) # 640
        frame_height = int(cap.get(4)) # 480


        if IsPose:
            lmList, world = detector.findPosition(img, draw=True)

            X = []
            Y = []
            Z = []
            for i in range(len(world)):
                X.append(world[i].x)
                Y.append(world[i].y)
                Z.append(world[i].z)
        
            ax.set_xlim3d(-0.7,0.7)
            ax.set_ylim3d(-0.7,0.7)
            ax.set_zlim3d(-0.7,0.7)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.scatter3D(np.array(X), np.array(Z), np.array(Y)*-1, color="green") 
            ax.plot3D((X[12],X[24]),(Z[12],Z[24]),(Y[12]*-1,Y[24]*-1),"red")
            ax.plot3D((X[11],X[23]),(Z[11],Z[23]),(Y[11]*-1,Y[23]*-1),"blue")
            fig.canvas.draw()
            imgFig = np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep = '')
            imgFig = imgFig.reshape(fig.canvas.get_width_height()[::-1]+(3,))
            imgFig = cv2.resize(imgFig,(200,200))

            imgFig = cv2.cvtColor(imgFig,cv2.COLOR_RGB2BGR)

            ax.cla()

            img[280:480,0:200,:] = imgFig



            #cv2.imshow("Plot",imgFig)

            #ax = fig.add_subplot(projection='3d')


            #ax.scatter(1, 1, 1, color="green")
            #ax.cla()
            #fig.show()


            if len(lmList) != 0:

                ## Joint angle of interest
                # Left Shoulder
                RightShoulderAng = detector.findAngle(img, 13, 11, 23)
                # Left Elbow
                RightElbowAng = detector.findAngle(img, 15, 13, 11)
                # Left Hip
                RightHipAng = detector.findAngle(img, 11, 23, 25)
                # Left Knee
                RightKneeAng = detector.findAngle(img, 23, 25, 27)

                # Right Shoulder
                LeftShoulderAng = detector.findAngle(img, 14, 12, 24)
                # Right Elbow
                LeftElbowAng = detector.findAngle(img, 16, 14, 12)
                # Right Hip
                LeftHipAng = detector.findAngle(img, 12, 24, 26)
                # Right Knee
                LeftKneeAng = detector.findAngle(img, 24, 26, 28)
                

            if status == 0:
                xRShoulder,yRShoulder = lmList[12][1:]
                xLShoulder,yLShoulder = lmList[11][1:]
                xRKnee,yRKnee = lmList[26][1:]
                xLKnee,yLKnee = lmList[25][1:]
                xRToe,yRToe = lmList[32][1:]
                xLToe,yLToe = lmList[31][1:]


                cv2.line(img,(xRShoulder,450),(xRShoulder-50,450),(255,0,0),3)
                cv2.line(img,(xLShoulder,450),(xLShoulder+50,450),(255,0,0),3)

                if np.abs(xRShoulder-25 - (xRToe)) <= 25:
                    cv2.circle(img,(xRToe,yRToe),10,(0,255,0),-1)
                    calib[0] = 1;
                elif xRToe < xRShoulder-25:
                    try:
                        img[yRToe-15:yRToe+15,xRToe-15:xRToe+15,:] = arrow1
                    except:
                        pass
                elif xRToe > xRShoulder:
                    try:
                        img[yRToe-15:yRToe+15,xRToe-15:xRToe+15,:] = arrow2
                    except:
                        pass

                if np.abs(xLShoulder+25 - (xLToe)) <= 25:
                    cv2.circle(img,(xLToe,yLToe),10,(0,255,0),-1)
                    calib[1] = 1;
                elif xLToe < xLShoulder:
                    try:
                        img[yLToe-15:yLToe+15,xLToe-15:xLToe+15,:] = arrow1
                    except:
                        pass
                elif xLToe > xLShoulder+25:
                    try:
                        img[yLToe-15:yLToe+15,xLToe-15:xLToe+15,:] = arrow2
                    except:
                        pass

                if np.sum(calib) == 2:
                    calibIdx += 1;
                    startTime = time.time()
                else:
                    calibIdx = 0
                    calib[0] = 0
                    calib[1] = 0
                    disKnee = []

                if calibIdx >= 1:
                    cv2.putText(img,"Stay there",(10,100),cv2.FONT_HERSHEY_PLAIN,3,(0,255,255),5)
                    disKnee.append(np.abs(xRKnee - xLKnee))

                if calibIdx >= 100:
                    status = 1;

            if status == 1:
                xRKnee,yRKnee = lmList[26][1:]
                xLKnee,yLKnee = lmList[25][1:]
                xRShoulder,_ = lmList[12][1:]
                xLShoulder,_ = lmList[11][1:]

                currentTime = time.time()

                countFrame += 1
                if (currentTime - startTime) >= 1 and (currentTime - startTime) <= 2:
                    cv2.putText(img, "3", (30,80), cv2.FONT_HERSHEY_PLAIN, 7, (0,255,0),5)
                
                if (currentTime - startTime) > 2 and (currentTime - startTime) <= 3:
                    cv2.putText(img, "2", (30,80), cv2.FONT_HERSHEY_PLAIN, 7, (0,255,0),5)

                if (currentTime - startTime) > 3 and (currentTime - startTime) <= 4:
                    cv2.putText(img, "1", (30,80), cv2.FONT_HERSHEY_PLAIN, 7,(0,255,0),5)

                if (currentTime - startTime) > 4 and (currentTime - startTime) <= 5:
                    cv2.putText(img, "Go!", (30,80), cv2.FONT_HERSHEY_PLAIN, 7,(0,255,0),5)

                threshKnee = np.mean(disKnee)

                currentDisKnee = np.abs(xRKnee-xLKnee)

                if currentDisKnee < threshKnee:
                    try:
                        img[yRKnee-15:yRKnee+15,xRKnee-15:xRKnee+15,:] = arrow2
                    except:
                        pass                
                    try:
                        img[yLKnee-15:yLKnee+15,xLKnee-15:xLKnee+15,:] = arrow1
                    except:
                        pass

                if xRShoulder < xRToe:
                    try:
                        img[yRShoulder-15:yRShoulder+15,xRShoulder-15:xRShoulder+15,:] = arrow1
                    except:
                        pass

                
                if xLShoulder > xLToe:
                    try:
                        img[yLShoulder-15:yLShoulder+15,xLShoulder-15:xLShoulder+15,:] = arrow2
                    except:
                        pass


                
                if RightKneeAng <= 90 and LeftKneeAng <= 90 and repDown == 0:
                    repDown = 1
                    repUp = 0

                if RightKneeAng >= 140 and LeftKneeAng >= 140 and repUp == 0:
                    repDown = 0
                    repUp = 1
                    rep += 1;

                if (currentTime - startTime) > 5:
                    cv2.putText(img, str(rep), (30,80), cv2.FONT_HERSHEY_PLAIN, 7, (0,255,255),5)


        else:
            continue

        countFrame += 1
        CountAngle = []

        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

        cv2.imshow('Image', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

    return redirect('/')