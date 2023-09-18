import mmcv
import tempfile
from mmtrack.apis import inference_sot, init_model

# 输入输出视频路径
input_video = 'data/bee.mp4'
output = 'results/SOT_bee.mp4'

# 指定单目标追踪算法 config 配置文件
sot_config = './configs/sot/siamese_rpn/siamese_rpn_r50_20e_lasot.py' # siamese_rpn
# sot_config = './configs/sot/stark/stark_st2_r50_50e_lasot.py' # stark_st2
# 指定单目标检测算法的模型权重文件
sot_checkpoint = 'https://download.openmmlab.com/mmtracking/sot/siamese_rpn/siamese_rpn_r50_1x_lasot/siamese_rpn_r50_1x_lasot_20211203_151612-da4b3c66.pth' # siamese_rpn
# sot_checkpoint = 'https://download.openmmlab.com/mmtracking/sot/stark/stark_st2_r50_50e_lasot/stark_st2_r50_50e_lasot_20220416_170201-b1484149.pth' # stark_st2
# 初始化单目标追踪模型
sot_model = init_model(sot_config, sot_checkpoint, device='cuda:0')

# 指定初始框的坐标 [x, y, w, h]
init_bbox = [132, 59, 57, 61]

# 转成 [x1, y1, x2, y2 ]
init_bbox = [init_bbox[0], init_bbox[1], init_bbox[0]+init_bbox[2], init_bbox[1]+init_bbox[3]]

# 读入待预测视频
imgs = mmcv.VideoReader(input_video)
prog_bar = mmcv.ProgressBar(len(imgs))
out_dir = tempfile.TemporaryDirectory()
out_path = out_dir.name
# 逐帧输入模型预测
for i, img in enumerate(imgs):
    result = inference_sot(sot_model, img, init_bbox, frame_id=i)
    
    # 绘制矩形框中心点构成的轨迹
    result_int = result['track_bboxes'].astype('uint32')
    
    sot_model.show_result(
            img,
            result,
            wait_time=int(1000. / imgs.fps),
            out_file=f'{out_path}/{i:06d}.jpg')
    prog_bar.update()

print(f'\n making the output video at {output} with a FPS of {imgs.fps}')
mmcv.frames2video(out_path, output, fps=imgs.fps, fourcc='mp4v')
out_dir.cleanup()