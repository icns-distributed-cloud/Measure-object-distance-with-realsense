from realsense_camera import *
import cv2
rs = RealsenseCamera()

while True:
    ret, bgr_frame, depth_frame = rs.get_frame_stream()  #ret : frame is true or false
    cv2.imshow("Bgr frame", bgr_frame)

    print(depth_frame)
    cv2.imshow("depth frame", depth_frame)      #depth frame : depth information matrix  pixel 단위 , 480 * 640
    key = cv2.waitKey(1)
    if key == 27:
        break
# cv2.imshow("Bgr frame", bgr_frame)
# cv2.waitKey(0)


