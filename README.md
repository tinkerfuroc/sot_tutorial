# sot_tutorial
There are some examples of sot algorithms in this repository

# 参考环境
Ubuntu

cuda

# 文件结构（斜体表示 .gitignore 的文件（夹），需要自行添加）
|——*data: 推荐的输入文件夹命名*

|　　|——*bee.mp4*: 推荐的示例视频，下载：`wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220418-mmtracking/data/bee.mp4 -O data/bee.mp4`

|——*results: 推荐的输出文件夹命名*

|——ltr: pytracking所需文件夹

|——pytracking

|　　|——pytracking_sot.py: 运行pytracking的示例代码，实现视频内跟踪，可在第36、37行更改想要的算法，输入输出路径可自定义

|　　|——*networks*

|　　　　|——*dimp50.pth: pytracking的模型，下载：`gdown https://drive.google.com/uc\?id\=1qgachgqks2UGjKx-GdO1qylBDdB1f9KN -O pytracking/networks/dimp50.pth`*

|　　　　|——*dimp18.pth: pytracking的模型，下载：`gdown https://drive.google.com/uc\?id\=1MAjrRJDCbL0DSjUKFyDkUuYS1-cYBNjk -O pytracking/networks/dimp18.pth`*

|　　　　|——*atom_default.pth: pytracking的模型，下载：`gdown https://drive.google.com/uc\?id\=1VNyr-Ds0khjM0zaq6lU-xfY74-iWxBvU -O pytracking/networks/atom_default.pth`*

|　　　　|——*resnet18_vggmconv1.pth: pytracking的模型，下载：`gdown https://drive.google.com/uc\?id\=1aWC4waLv_te-BULoy0k-n_zS-ONms21S -O pytracking/networks/resnet18_vggmconv1.pth`*

|　　|——……

|——getROI.py: 运行后手动框选ROI，按空格后即可打印出ROI的坐标(x, y, w, h)

|——mmtracking_sot.py: 运行mmtracking的示例代码，实现视频跟踪，可在第 10-14 行选择想要的算法（siamese_rpn或stark_st2），可在第19行更改初始框选坐标，输入输出路径可自定义

|——opencv_sot_camera.py: 运行opencv的示例代码，实现摄像头内跟踪，可在第24行更改想要的算法，输入输出路径可自定义

|——opencv_sot.py: 运行opencv的示例代码，实现视频内跟踪，可在第20行更改想要的算法，输入输出路径可自定义

# OpenCV

```bash
conda create --name opencv python=3.7
conda activate opencv
pip install opencv-python
python opencv_sot.py    // 按空格开始播放，再按空格会结束，并把视频保存到results文件夹
python opencv_sot_camera.py // 按空格开始播放，再按Q会结束，并把视频保存到results文件夹
```
# mmtracking

```bash
conda create --name mmtracking python=3.7
conda activate mmtracking

# 在pytorch官网上找到对应电脑cuda版本的torch版本，这里仅供参考
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# 安装 mmcv-full1.7.1，不要用 mmcv2.0.0rc4，地址中的cu和torch根据自己的版本来替换
pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.31.1/index.html

pip install mmdet==2.28.2
pip install mmtrack==0.14.0

pip install seaborn ipywidgets tqdm imageio-ffmpeg ninja -i https://pypi.tuna.tsinghua.edu.cn/simple

python mmtracking_sot.py    // 按空格开始播放，再按空格会结束，并把视频保存到results文件夹
```

# pytracking

```bash
conda create --name pytracking python=3.7
conda activate pytracking

# 在pytorch官网上找到对应电脑cuda版本的torch版本，这里仅供参考
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

pip install matplotlib pandas tqdm opencv-python visdom tb-nightly scikit-image tikzplotlib gdown ninja jpeg4py

python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"

# Download the default network for DiMP-50 and DiMP-18
gdown https://drive.google.com/uc\?id\=1qgachgqks2UGjKx-GdO1qylBDdB1f9KN -O pytracking/networks/dimp50.pth
gdown https://drive.google.com/uc\?id\=1MAjrRJDCbL0DSjUKFyDkUuYS1-cYBNjk -O pytracking/networks/dimp18.pth

# Download the default network for ATOM
gdown https://drive.google.com/uc\?id\=1VNyr-Ds0khjM0zaq6lU-xfY74-iWxBvU -O pytracking/networks/atom_default.pth

# Download the default network for ECO
gdown https://drive.google.com/uc\?id\=1aWC4waLv_te-BULoy0k-n_zS-ONms21S -O pytracking/networks/resnet18_vggmconv1.pth

python pytracking/pytracking_sot.py     // 按空格开始播放，再按空格会结束，并把视频保存到results文件夹
```