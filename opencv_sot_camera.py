import cv2
import time

# 设置参数
video_save_path = r'results/camera2.mp4'  	# 视频存放地址


# 调用摄像头获取帧
cap = cv2.VideoCapture(1) # Mac电脑的参数为1，Windows电脑的参数为0

cap.open(0)
time.sleep(1)
ret, frame = cap.read()							# 读取帧图像


frame_width = int(cap.get(3))					# 获取图像宽,并转换为整数	
frame_height = int(cap.get(4))					# 获取图像高,并转换为整数	
# 创建保存视频的对象（设置编码格式，帧率，图像的宽高等）
out = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), 30, (frame_width, frame_height))

bbox = cv2.selectROI('frame', frame, fromCenter=False, showCrosshair=True)	
# init_bbox = (132, 59, 57, 61)
# 人工标注感兴趣目标
tracker = cv2.legacy.TrackerCSRT_create()		# 使用csrt算法
tracker.init(frame, bbox)						# 初始化tracker

while cap.isOpened():
    success, frame = cap.read()						# 读取帧图像
    if frame is None:							# 如果读到的帧数不为空，则继续读取；如果为空，则退出。
        break

    if not success:
        print('Error')
        break
        
    bool_para, box = tracker.update(frame)		# 由于物体运动，需要动态的根据物体运动更新矩形框
    if bool_para:								# 若读取成功，我们就定位画框，并跟随
        (x, y, w, h) = [int(ii) for ii in box]
        cv2.rectangle(frame, pt1=(int(x),int(y)), pt2=(int(x)+int(w), int(y)+int(h)), color=(0, 255, 0), thickness=2)
        cv2.imshow('camera', frame)
        out.write(frame)            			# 将每一帧图像写入到输出文件中

    if cv2.waitKey(1) in [ord('q'),27]: # 按下键盘的 q 或 esc 退出（在英文输入法下）
        break
    
cap.release()				# 释放摄像头
out.release()				# 释放摄像头
cv2.destroyAllWindows()		# 摧毁所有图窗