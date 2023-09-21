# Face_beauty
 使用传统算法对人脸美颜，用到计算机视觉知识

 ## 1. 人脸检测
 借助opencv中的CascadeClassifier函数

 ## 2. 滤波器
 ### 2.1 引导滤波
 ![image](https://github.com/yxyxnrh/Face_beauty/assets/82510221/6f3dc2ea-6fac-484b-b9da-f83eb9e557bb)
 ### 2.2 NLM滤波

大窗口是以目标像素为中心的搜索窗口，两个灰色小窗口分别是以x, y为中心的邻域窗口。其中以y为中心的邻域窗口在搜索窗口中滑动，通过计算两个邻域窗口间的相似程度为y赋以权值
 ![image](https://github.com/yxyxnrh/Face_beauty/assets/82510221/79886efd-57f5-494c-b33b-83d937758c59)

 
引导滤波图  
  ![image](https://github.com/yxyxnrh/Face_beauty/assets/82510221/6e9c5978-41eb-47cb-8bcd-38fe45c4ef06)
自定义NLM图
  ![image](https://github.com/yxyxnrh/Face_beauty/assets/82510221/39d07952-0181-4d1c-abae-4b83f271792f)

 ### 3.皮肤检测
 满足以下式子：
 R>95 And G>40 And B>20 And R>G And R>B And Max(R,G,B)-Min(R,G,B)>15 And Abs(R-G)>15

 ## 4. 融合
1）对原始图像进行预处理和图像配准；
2）对处理过的图像分别进行小波分解，得到低频和高频分量；
3）对低频和高频分量采用不同的融合规则进行融合；
4）进行小波逆变换；
5）得到融合图像。

## 5. 锐化
使用Gabor滤波来提取图像特征，锐化图像

## 6 结果
| 图片标签 | 图片 |
|----------|------|
| `<img src="image1.jpg" alt="![image](https://github.com/yxyxnrh/Face_beauty/assets/82510221/6e658bf3-8382-46a1-b949-3e88423c3495)
" width="400" height="300">` | ![Image 1](image1.jpg) |
| `<img src="image2.jpg" alt="![image](https://github.com/yxyxnrh/Face_beauty/assets/82510221/65d194ee-7eb2-4bb4-8e1e-9a015c232786)
" width="400" height="300">` | ![Image 2](image2.jpg) |






