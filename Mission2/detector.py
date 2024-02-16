import sys, os
# darknet 라이브러리 설정
sys.path.append(os.path.join(os.getcwd(), '/home/ddwu/opencv/opencv-4.4.0/build/darknet/python'))

import cv2
import darknet as dn
import pdb

# darknet 라이브러리 설정
sys.path.append(os.path.join(os.getcwd(), '/home/ddwu/opencv/opencv-4.4.0/build/darknet'))
dn.set_gpu(0)
net = dn.load_net(b"/home/ddwu/opencv/opencv-4.4.0/build/darknet/cfg/yolov3.cfg", b"/home/ddwu/opencv/opencv-4.4.0/build/darknet/yolov3.weights", 0)
meta = dn.load_meta(b"/home/ddwu/opencv/opencv-4.4.0/build/darknet/cfg/coco.data")

# And then down here you could detect a lot more images like:
r = dn.detect(net, meta, b"data/eagle.jpg")
print(r)
r = dn.detect(net, meta, b"data/giraffe.jpg")
print(r)
r = dn.detect(net, meta, b"data/horses.jpg")
print(r)
r = dn.detect(net, meta, b"data/person.jpg")
print(r)
