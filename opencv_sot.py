import cv2

# 设置参数
video_load_path = r'data/bee.mp4'  	# 视频存放地址
video_save_path = r'results/bee2.mp4'  	# 视频存放地址


cap = cv2.VideoCapture(video_load_path)			# 读取视频流
ret, frame = cap.read()							# 读取帧图像


frame_width = int(cap.get(3))					# 获取图像宽,并转换为整数	
frame_height = int(cap.get(4))					# 获取图像高,并转换为整数	
# 创建保存视频的对象（设置编码格式，帧率，图像的宽高等）
out = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), 30, (frame_width, frame_height))

bbox = cv2.selectROI('frame', frame, fromCenter=False, showCrosshair=True)
# bbox = (132, 59, 57, 61)		
# 人工标注感兴趣目标
tracker = cv2.legacy.TrackerCSRT_create()		# 使用csrt算法
tracker.init(frame, bbox)						# 初始化tracker

while True:
    _, frame = cap.read()						# 读取帧图像
    if frame is None:							# 如果读到的帧数不为空，则继续读取；如果为空，则退出。
        break
        
    bool_para, box = tracker.update(frame)		# 由于物体运动，需要动态的根据物体运动更新矩形框
    if bool_para:								# 若读取成功，我们就定位画框，并跟随
        (x, y, w, h) = [int(ii) for ii in box]
        cv2.rectangle(frame, pt1=(int(x),int(y)), pt2=(int(x)+int(w), int(y)+int(h)), color=(0, 255, 0), thickness=2)
        cv2.imshow('frame', frame)				# 实时显示追求目标
        out.write(frame)            			# 将每一帧图像写入到输出文件中
    
    # 使用 waitKey 可以控制视频的播放速度。数值越小，播放速度越快。
    if cv2.waitKey(1) == ord(' '):	# ord(' '): 按空格结束
        break
    
cap.release()				# 释放摄像头
out.release()				# 释放摄像头
cv2.destroyAllWindows()		# 摧毁所有图窗