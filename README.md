本项目使用python编程，实现车牌的自动识别系统

传统的车牌识别过程如下

照片->预处理->车牌定位->字符分割->字符识别->车牌号码

而本项目采用深度学习的方式，去除了字符分割这一步骤

训练的数据集为CCPD https://github.com/detectRecog/CCPD

对于车牌定位，选用了yolov5模型 https://github.com/ultralytics/yolov5

对于字符识别，选用了CRNN（CNN+RNN+CTCloss）模型  车牌监测正确率：95.9% 字符识别正确率：99.1%


scrips中包括了各类脚本，自动化处理数据，方便模型训练、测试以及性能评估

执行test.py用于测试整个系统的性能
输出
场景：xxx  准确率：xx%  FPS（每秒处理图片数）：xxx

最终测试结果，整体准确率达到82.44%，其中基础场景正确率达到91.48%


执行main.py启动gui界面，可以选择图片进行识别


Q：为什么选用yolov5呢？

A：简单易部署，适合初学者

Q：为什么选用CRNN呢？

A：对于车牌号而言，传统的蓝牌由 省份简称+市级字母+5位字母或数字 组成，而系能源车牌则要在传统蓝牌的基础上，加上1位字母或数字。由于CRNN能够识别不定长度的字符，我们仅需要训练一个模型，就能够同时识别传统车牌以及新能源车牌的号码。同时，CRNN模型的实现难度适中，对于初学者来说很方便。
