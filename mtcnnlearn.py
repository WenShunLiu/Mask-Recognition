import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torch_py.Utils import plot_image
from torch_py.MTCNN.detector import FaceDetector
from torch_py.MobileNetV1 import MobileNetV1
from torch_py.FaceRec import Recognition

pnet_path = "./torch_py/MTCNN/weights/pnet.npy"
rnet_path = "./torch_py/MTCNN/weights/rnet.npy"
onet_path = "./torch_py/MTCNN/weights/onet.npy"

img = Image.open("image/test.jpg")
detector = FaceDetector()
recognize = Recognition(model_path='./results/modelV1.pkl')
draw, all_num, mask_nums = recognize.mask_recognize(img)

print("总人数:", all_num, "戴口罩数", mask_nums)
plt.title("total: %d,  mask count: %d" %(all_num, mask_nums)) 
plt.imshow(draw)
plt.show()
