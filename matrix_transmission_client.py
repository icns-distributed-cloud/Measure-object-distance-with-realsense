import socket 
import numpy as np
from realsense_camera import *
import cv2
from queue import Queue
from _thread import *
import pickle

enclosure_queue1 = Queue()
enclosure_queue2 = Queue()

rs = RealsenseCamera()



def data_transmission(queue1,queue2):
    while True:
        ret, bgr_frame, depth_frame = rs.get_frame_stream()  #ret : frame is true or false
        if ret == False:
            continue

        #depth
        data_string = pickle.dumps(depth_frame)
        queue1.put(data_string)

        #bgr
        encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
        result, imgencode = cv2.imencode('.jpg', bgr_frame, encode_param) 
        data = np.array(imgencode)
        stringData = data.tostring()
        queue2.put(stringData)


        cv2.imshow("depth frame Client", depth_frame)
        # print(depth_frame)
        
        cv2.imshow("bgr frame Client", bgr_frame)      #depth frame : depth information matrix  pixel 단위 , 480 * 640
        key = cv2.waitKey(1)
        if key == 27:
            break

HOST = '127.0.0.1'             #호스트는 16x.xxx.xxx.xx 학교 서버 주소
PORT = 9999

client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM) 

client_socket.connect((HOST, PORT))

start_new_thread(data_transmission, (enclosure_queue1,enclosure_queue2))
while True: 

    message = '1'
    client_socket.send(message.encode()) #encode : 문자열을 byte로 변환해줌 

    client_socket.recv(1024) #2 받아줌

    depthStringData = enclosure_queue1.get()
    bgrStringData = enclosure_queue2.get()
    client_socket.send(str(len(depthStringData)).ljust(16).encode()) # 데이터 수신 확인. recv함수는 수신될 데이터의 크기를 미리 알아야 하기 때문에 서버에서 전송할 이미지의 크기를 보내서
    client_socket.send(depthStringData)   
    client_socket.send(str(len(bgrStringData)).ljust(16).encode()) 
    client_socket.send(bgrStringData)                          # 클라이언트에서 수신받을 준비를 하게 하고 이미지를 전송한다.


client_socket.close() 