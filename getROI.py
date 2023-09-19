# opencv-python
import cv2

# 导入python绘图matplotlib
import matplotlib.pyplot as plt

# 定义可视化图像函数
def show_img_from_array(img):
    '''opencv读入图像格式为BGR，matplotlib可视化格式为RGB，因此需将BGR转RGB'''
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_RGB)
    plt.show()

# 获取视频第一帧
input_path = './data/bee.mp4'
cap = cv2.VideoCapture(input_path)
if cap.isOpened():
    success, frame = cap.read()
cap.release()

r = cv2.selectROI("initial bbox", frame)

print(r)