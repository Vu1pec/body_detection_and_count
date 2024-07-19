这个项目主要难点在于通过我们自己训练的随机森林模型对动作进行分类，计数是其次的

需要优化的还有（分类）：用户乱动/idle状态的识别过滤。模型准确率（其实还行，只要正对摄像头）。

需要优化的还有（计数）：计数优化（一些特殊情况可能导致计数器不工作：比如你引体向上拉得太慢或者不标准（只有脖子在动是不行的，，，））

需要优化的还有（代码）：代码很乱，改参数变量不方便。readme写的一坨。

运行main.py就可以直接开始程序了，调整84行的VideoCapture函数来选择不同视频/摄像头
self.cap = cv2.VideoCapture(0)  # 设置为0就可以变成摄像头，mac注意在设置里给pycharm权限

如果视频比例看起来不正常，可能会影响检测准确度，请调整 main.py 22、23行的self.width和self.height

对于视频识别，如果帧序列处理过慢，可能需要降低分辨率
同样对于摄像头，如果处理速度过慢，可能会跳过帧信息，建议降低分辨率

使用main.py时请注意要有scikit-learn（conda_base已经内置）
内置了两个模型，分别为

(label_encoder.joblib, action_recognition_model.joblib)
(label_encoder2.joblib, action_recognition_model2.joblib)

如果要更改，在main.py的main入口（最下面）还有53行的load_model函数更改名字。

如果要自己训练模型，先通过features.py在main入口的video_path选择训练视频。

features.py处理后会在同目录创建一个features文件夹，里面有csv文件

annotations同理，键盘操作复杂，建议手搓json

采用如下结构：[{"action": "jump_rope", "start_frame": 0, "end_frame": 240}, {...}, ...]

annotations.py会在同目录创建annotations文件夹，里面有json文件

最后用train.py，在main入口更改.csv/.json路径，修改输出的模型命名。

比如：action_recognition_model3.joblib ｜ label_encoder3.joblib

训练模型注意事项：每种动作的帧数不能相差太大。比如100帧跳绳，100帧俯卧撑。

如果对侧着跳绳的识别不佳，可以增加100帧的侧着跳绳，注意建议让模型当作一种新的动作，比如'jump_rope_side'

你可以修改standard_pull/push_up.csv设定标准的动作（这里存储了关键帧动作，注意是13*4=52个关键点，与features.csv的每一行（除了frame列）的个数相同）
