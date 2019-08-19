# vgg_face_caffe
人脸识别简要说明

> face_recognition.py，利用OpenCv自带人脸检测器检测人脸，然后利用VGGNet提取人脸图像特征，求注册图像和待识别图像的余弦相似度来判断是否是同一个人。
程序有两个分支用于判断是注册阶段或者是识别阶段。输入1是注册,输入2是识别,我们一般注册五张图片,识别的时候用当前的人脸和这五张比较,取平均值，当该平均值大于设定阈值，则为同一个人，识别成功；反之，则识别失败。

vgg_face.caffemodel百度云盘下载链接：https://pan.baidu.com/s/1Kf3zUiM70awxG2PtGG8Nrw 
提取码：u6m7 
