import cv2
import importlib
import os
from collections import OrderedDict
from pathlib import Path
import numpy as np
import sys

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import Tracker
from pytracking.evaluation.multi_object_wrapper import MultiObjectWrapper

_tracker_disp_colors = {1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 0, 0),
                        4: (255, 255, 255), 5: (0, 0, 0), 6: (0, 255, 128),
                        7: (123, 123, 123), 8: (255, 128, 0), 9: (128, 0, 255)}

def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module('pytracking.parameter.{}.{}'.format(self.name, self.parameter_name))
        params = param_module.parameters()
        return params

# 设置参数
videofilepath = r'data/bee.mp4'  	# 视频存放地址
video_save_path = r'results/bee1.mp4'
results_dir = r'results'  	# 视频存放地址


# bbox = cv2.selectROI('frame', frame, fromCenter=False, showCrosshair=True)	
init_bbox = [132, 59, 57, 61]
# 人工标注感兴趣目标

tracker_name = 'dimp'
tracker_param = 'dimp50'
trackerClass = Tracker(tracker_name, tracker_param)
params = trackerClass.get_parameters()
params.tracker_name = tracker_name
params.param_name = tracker_param

multiobj_mode = getattr(params, 'multiobj_mode', getattr(trackerClass.tracker_class, 'multiobj_mode', 'default'))
if multiobj_mode == 'default':
    tracker = trackerClass.create_tracker(params)
    if hasattr(tracker, 'initialize_features'):
        tracker.initialize_features()
elif multiobj_mode == 'parallel':
    tracker = MultiObjectWrapper(trackerClass.tracker_class, params, trackerClass.visdom, fast_load=True)
else:
    raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))


frame_number = 0

if videofilepath is not None:
    assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
    ", videofilepath must be a valid videofile"
    cap = cv2.VideoCapture(videofilepath)
    ret, frame = cap.read()
    frame_number += 1
else:
    cap = cv2.VideoCapture(0)
next_object_id = 1
sequence_object_ids = []
prev_output = OrderedDict()
output_boxes = OrderedDict()
frame_width = int(cap.get(3))					# 获取图像宽,并转换为整数	
frame_height = int(cap.get(4))					# 获取图像高,并转换为整数	
# 创建保存视频的对象（设置编码格式，帧率，图像的宽高等）
result = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), 30, (frame_width, frame_height))

bbox = [132.0, 59.0, 57.0, 61.0]
out = tracker.initialize(frame, {'init_bbox': OrderedDict({next_object_id: bbox}),
                                'init_object_ids': [next_object_id, ],
                                'object_ids': [next_object_id, ],
                                'sequence_object_ids': [next_object_id, ]})
prev_output = OrderedDict(out)

output_boxes[next_object_id] = [bbox, ]
sequence_object_ids.append(next_object_id)
next_object_id += 1

new_init = True
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_number += 1
    if frame is None:
        break
    frame_disp = frame.copy()
    info = OrderedDict()
    info['previous_output'] = prev_output

    if new_init:
        new_init = False
        init_state = bbox

        info['init_object_ids'] = [next_object_id, ]
        info['init_bbox'] = OrderedDict({next_object_id: init_state})
        sequence_object_ids.append(next_object_id)
        output_boxes[next_object_id] = [init_state, ]
        next_object_id += 1
    
    if len(sequence_object_ids) > 0:
        info['sequence_object_ids'] = sequence_object_ids
        out = tracker.track(frame, info)
        prev_output = OrderedDict(out)
    
        if 'target_bbox' in out:
            for obj_id, state in out['target_bbox'].items():
                state = [int(s) for s in state]
                cv2.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                                _tracker_disp_colors[obj_id], 5)
                output_boxes[obj_id].append(state)
                result.write(frame_disp)
    
cap.release()				# 释放摄像头
cv2.destroyAllWindows()		# 摧毁所有图窗

if not os.path.exists(results_dir):
    os.makedirs(results_dir)
video_name = "webcam" if videofilepath is None else Path(videofilepath).stem
base_results_path = os.path.join(results_dir, 'video_{}'.format(video_name))
print(f"Save results to: {base_results_path}")
for obj_id, bbox in output_boxes.items():
    tracked_bb = np.array(bbox).astype(int)
    bbox_file = '{}_{}.txt'.format(base_results_path, obj_id)
    np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')
