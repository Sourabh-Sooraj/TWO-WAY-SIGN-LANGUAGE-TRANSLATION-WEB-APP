from os import name
from flask import Flask, render_template, request, Response
import cv2, imutils, time
import pyshine as ps
import pygame
import numpy as np
from sklearn.metrics import pairwise
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

background=None

accumulated_weight = 0.5

roi_top=20
roi_bottom=300
roi_right=300
roi_left=600

#FINDING AVERAGE BACKGROUND VALUE
def calc_accum_avg(frame, accumulated_weight):
    
    global background
    
    if background is None:
        background=frame.copy().astype('float')
        return None
    
    cv2.accumulateWeighted(frame, background, accumulated_weight)
    
def segment(frame, threshold_min=25):
    
    diff= cv2.absdiff(background.astype('uint8'),frame)
    
    ret,thresholded = cv2.threshold(diff, threshold_min, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    image, contours, heirarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours)==0:
        return None
    else:
        #ASSUMING THE LARGEST EXTERNAL CONTOUR IN THE ROI, IS THE HAND
        hand_segment = max(contours, key=cv2.contourArea)
        
        return (thresholded, hand_segment)
    
def sign_language(thresholded, hand_segment):
    conv_hull = cv2.convexHull(hand_segment)
    
    #TOP
    top=tuple(conv_hull[conv_hull[:,:,1].argmin()][0])
    bottom=tuple(conv_hull[conv_hull[:,:,1].argmax()][0])
    left=tuple(conv_hull[conv_hull[:,:,0].argmin()][0])
    right=tuple(conv_hull[conv_hull[:,:,0].argmax()][0])
    
    cX = (left[0]+right[0]) // 2
    cY = (top[1]+bottom[1]) // 2
    
    distance = pairwise.euclidean_distances([(cX,cY)], Y=[left,right,top,bottom])[0]
    
    max_distance = distance.max()
    
    radius=int(0.67*max_distance)
    circumfrence = (2*np.pi*radius)
    
    circular_roi=np.zeros(thresholded.shape[:2],dtype='uint8')
    
    cv2.circle(circular_roi, (cX,cY), radius, 255,10)
    
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)
    
    image,contours,heirarchy = cv2.findContours(circular_roi.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    count=0 
    
    for cnt in contours:
        (x,y,w,h) = cv2.boundingRect(cnt)
        out_of_wrist = ((cY+(cY*0.25))>(y+h))
        
        limit_points = ((circumfrence*0.35)>cnt.shape[0])
        
        fing_dist_x = cX+(cX*0.5)
        
        fing_dist_y = cY+(cY*0.5)
        
        letter='NIL'
        
        if out_of_wrist and limit_points:
            count+=1
            
            if ((count==1) and (max_distance>70 and max_distance<135) and (circumfrence>330 and circumfrence<560)):
                letter='S'
                     
            if (max_distance>150) and (circumfrence>610):
                letter='Y'
                
            if ((count==2) and (max_distance>125 and max_distance<145) and (circumfrence<560)):
                letter='B'
                
            if ((count==1) and (max_distance>150) and (circumfrence>600)):
                letter='C'
                
            if ((count==2) and (max_distance>110 and max_distance<115) and (circumfrence>470 and circumfrence<475)):
                letter='D'
                
            if ((count==3) and (max_distance>147 and max_distance<157) and (circumfrence>640 and circumfrence<660)):
                letter='F'
                
            if ((count==1) and (max_distance>140 and max_distance<155) and (circumfrence>600 and circumfrence<645)):
                letter='G'
                
            if ((count==1) and (max_distance>155 and max_distance<170) and (circumfrence>650 and circumfrence<705)):
                letter='H'
                
            if ((count==1) and (max_distance>140 and max_distance<150) and (circumfrence>620 and circumfrence<630)):
                letter='I'
                
            if ((count==2) and (max_distance>130 and max_distance<140) and (circumfrence>550 and circumfrence<560)):
                letter='K'
                
            if ((count==1) and (max_distance>145 and max_distance<150) and (circumfrence>550 and circumfrence<620)):
                letter='L'
                
            if ((count==1) and (max_distance>110 and max_distance<125) and (circumfrence>428 and circumfrence<500)):
                letter='N'
                
            if ((count==1) and (max_distance>135 and max_distance<140) and (circumfrence>540 and circumfrence<615)):
                letter='P'
                
            if ((count==1) and (max_distance>140 and max_distance<150) and (circumfrence>570 and circumfrence<610)):
                letter='Q'
                
            if ((count==1) and (max_distance>125 and max_distance<130) and (circumfrence>520 and circumfrence<545)):
                letter='R'
                
            if ((count==2) and (max_distance>135 and max_distance<150) and (circumfrence>570 and circumfrence<628)):
                letter='T'
                
            if ((count==1) and (max_distance>125 and max_distance<150) and (circumfrence>525 and circumfrence<540)):
                letter='U'
                
            if ((count==2) and (max_distance>115 and max_distance<125) and (circumfrence<535)):
                letter='V'
                
            if ((count==3) and (max_distance>115 and max_distance<135) and (circumfrence>555 and circumfrence<560)):
                letter='W'
                
            if ((count==1) and (max_distance>125 and max_distance<135) and (circumfrence>520 and circumfrence<560)):
                letter='X'          

        elif ((count==0) and (max_distance>108 and max_distance<115) and (circumfrence>470 and circumfrence<480)):
            letter='E'
            
        elif ((count==0) and (max_distance>118 and max_distance<142) and (circumfrence>480 and circumfrence<580)):
            letter='M'
            
        elif ((count==0) and (max_distance>143 and max_distance<152) and (circumfrence>600 and circumfrence<610)):
            letter='O'
            
        elif ((count==0) and (max_distance>50 and max_distance<105) and (circumfrence>390 and circumfrence<450)):
            letter='A'
            
            
    return letter+" "+str(count)+" "+str(max_distance)+" "+str(circumfrence)

def signlang():
    
    # CAMERA=False
    # if not CAMERA:
    #     print('ERROR')
    # else:
        cam=cv2.VideoCapture(0)

        num_frames=0

        timer = 0

        f= open("guru99.txt","w+")

        while True:
            ret, frame = cam.read()
            
            frame=cv2.flip(frame,1)
            
            frame_copy = frame.copy()
            
            roi=frame[roi_top:roi_bottom,roi_right:roi_left]
            
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            gray=cv2.GaussianBlur(gray, (7,7), 0)
            
            if num_frames<60:
                calc_accum_avg(gray, accumulated_weight)
                
                if num_frames<=59:
                    cv2.putText(frame_copy, 'WAIT, GETTING BACKGROUND',(200,400),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                    # cv2.imshow('Finger Count', frame_copy)
            else:
                hand=segment(gray)
                    
                if hand is not None:
                        
                    thresholded, hand_segment = hand
                        
                    #DRAWS CONTOURS AROUND REAL HAND IN THE LIVESTREAM
                    cv2.drawContours(frame_copy, [hand_segment+(roi_right,roi_top)],-1,(255,0,0),1)
                        
                    letters = sign_language(thresholded, hand_segment)
                    
                    if timer%150==0:
                        cv2.waitKey(1)
                        f.write("%s" % letters.split(" ")[0])
                        
                        if letters.split(" ")[0]!='NIL':
                            pygame.mixer.init()
                            pygame.mixer.music.load("static/Voice_recs/"+letters.split(" ")[0]+".mp3")
                            pygame.mixer.music.play()
                        
                        
                    cv2.putText(frame_copy,str(letters),(70,45),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                        
                    # cv2.imshow('Thresholded', thresholded)
                        
            cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0,0,255), 5)
                    
            num_frames+=1
                    
            finframe=cv2.imencode('.JPEG',frame_copy)[1].tobytes()
            timer=timer+1
                
            # k=cv2.waitKey(1) & 0xFF
                
            # if k==27:
            #     break
            
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n'+finframe+b'\r\n')


# def scanning(sent):
#     for x in sent:
#         capturing(x)
        
    
# def capturing(sent):
    
#     print(sent)
    
#     for x in sent:
        
#         cap = cv2.VideoCapture('D:/ComputerVisionCourse/07-Capstone-Project/Video recs/'+x+'.mp4')

#         if cap.isOpened()==False:
#             print("ERROR OPENING FILE OR WRONG CODEC USED")

#         while cap.isOpened():

#             ret, frame = cap.read()

#             if ret == True:

#                 if x=='J' or x=='K' or x=='L' or x=='M' or x=='N' or x=='O' or x=='P' or x=='Q' or x=='R':
#                     frame=cv2.resize(frame,(720, 405),fx=0,fy=0,interpolation=cv2.INTER_AREA)
#                     #WRITER 80 FPS
#                     time.sleep(1/80)
#                 else:
#                     #WRITER 45 FPS
#                     time.sleep(1/45)

#                 fr = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
#                 finframe=cv2.imencode('.JPEG',fr)[1].tobytes()
                
#                 yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n'+finframe+b'\r\n')

            #     if cv2.waitKey(1) & 0xFF == ord('q'): 
            #         break
            # else:
            #     break

    # cap.release()
    # cv2.destroyAllWindows()

# @app.route('/res',methods=['POST','GET'])
# def res():
#     global result
#     if request.method == 'POST':
#         result=request.form.get('sent')
#         return render_template('results.html')
    
# @app.route('/vid_feed')
# def vid_feed():
#     print(result)
#     return Response(capturing(result),mimetype='multipart/x-mixed-replace; boundary=frame')

    
# @app.route('/vid_feed_2', methods=['GET'])
# def vid_feed_2():
#     sentence = request.args.get('sentence')
#     print(sentence)
#     return Response(capturing(sentence),mimetype='multipart/x-mixed-replace; boundary=frame')




@app.route('/signlang')
def vidtotxt():
     return render_template('signlang.html')
 
 
@app.route('/video')
def video():
     return Response(signlang(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     # Threaded option to enable multiple instances for multiple user access support
#     app.run(threaded=True, port=5000)

if __name__ == "__main__":
    app.run(debug=True, port=5000)