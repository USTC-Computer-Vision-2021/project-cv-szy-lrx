[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-f059dc9a6f8d3a56e377f745f24479a46679e63a5d9fe6f495e02850cd0d8118.svg)](https://classroom.github.com/online_ide?assignment_repo_id=6405850&assignment_repo_type=AssignmentRepo)

基于图像拼接的 A Look into the Past
====

成员及分工
* 宋子杨 PB18061254
  * 调研、编程、写报告
* 刘润昕 PB18061277
  * 调研、编程、写报告

问题描述
---
> **闲云潭影日悠悠，物换星移几度秋**

- 项目灵感：用两张不同时期的小屋照片实现A Look into the Past, 让平凡的景物见证岁月的变迁。
- 背景知识：图像拼接，包含图像特征点匹配、图像重投影、图像拼接、图像融合等具体步骤。

实验原理
---
图像拼接在运动相机检测和跟踪、增强现实、分辨率增强、视频压缩和图像稳定等机器视觉领域有很大的应用，它主要包括四个步骤：图像匹配、重投影、缝合和融合。
* 图像匹配：描述相同场景一组图片之间的几何对应关系，这组图片可以是不同时间不同位置的拍摄，也可以是由多个传感器同时拍摄的多张图片。
* 重投影：通过图像几何变换，把一系列图片转换成一个共同的坐标系。
* 缝合：通过合并重叠部分的像素值并保持没有重叠的像素值使之生成更大画布的图像。
* 融合：通过几何和光度偏移错误通常导致对象的不连续，并在两个图像之间的边界附近产生可见的接缝。因此，为了减小接缝的出现，需要在缝合时或缝合之后使用混合算法。

本实验的大致流程如下图所示：
![](https://github.com/USTC-Computer-Vision-2021/project-cv-szy-lrx/blob/main/USED-Image/hw_1.png)
### 1.特征提取与匹配
特征提取采用SIFT算法，提取与匹配阶段包括以下三个步骤：
* 检测尺度空间极值，精确定位关键点：搜索所有尺度上的图像位置，通过高斯微分函数来识别对于尺度和旋转不变的兴趣点，拟合三维二次函数来精确确定关键点的位置。
* 为关键点分配主方向，进行特征描述：基于局部的梯度方向，分配给每个关键点位置一个或多个方向，计算周围图像块内梯度直方图，生成描述子。
* 特征匹配：通过检测特征点之间的距离来寻找合适的关键点匹配。

SIFT算法具有旋转不变性及尺度不变性，非常适合于高分辨率图像中的目标检测。

### 2.单应矩阵计算
有了两组相关点，接下来就需要建立两组点的转换关系，也就是图像变换关系。单应性是两个空间之间的映射，常用于表示同一场景的两个图像之间的对应关系，可以匹配大部分相关的特征点，并且能实现图像投影，使一张图通过投影和另一张图实现大面积的重合。在本实验中，我们使用RANSAC算法计算单应矩阵，它可以鲁棒地估计模型参数，找到于大多数点相关地单应矩阵，并将不正确地匹配作为异常值丢弃。

### 3.图像变形与融合
使用2.中获得的单应性矩阵对图像进行变形，将右侧图片映射到结果图上，在进行融合时，我们选用了两种方法，Poisson Blending以及Alpha Blending。
* Poisson Blending：通过待嵌入图像、目标图像、目标插入区域、目标图像中位置-像素映射关系，求解关于融合后图像的梯度的变分方程，使得融合后的图像与原图像梯度一致，并且在嵌入区域的边界保持融合前后像素一致，实现无缝自然的图像融合。
* Alpha Blending：使用待嵌入图像大小的掩膜图像，对其进行变形，然后对变形后的图像进行羽化，再乘以对应图像，与加权后的背景图像相加得到最终结果。

代码实现
---
### 1.特征提取
```
#使用SIFT算法进行特征提取，返回特征点坐标及特征描述
def FeatureExtract(image):
    descriptor = cv2.SIFT_create()
    kps, des = descriptor.detectAndCompute(image, None)
    #将返回的特征点转化为numpy格式的数组
    kps = np.float32([kp.pt for kp in kps])
    return kps, des
```
### 2.特征匹配
```
#获取好的匹配关系，取比率为0.75
def FeatureMatch(des1, des2):
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k = 2)
    matches = sorted(matches, key = lambda x: x[0].distance / x[1].distance)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good
 ``` 
 ### 3.图像拼接与融合（Poisson Blending）
 ```
 def StitchAndBlend(image1, image2):
    kps1, des1 = FeatureExtract(image1)
    kps2, des2 = FeatureExtract(image2)
    matches = FeatureMatch(des1, des2)
    # 当筛选项的匹配对大于4对时：计算视角变换矩阵
    if len(matches) > 4:
        pts1 = np.float32([kps1[m.queryIdx] for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kps2[m.trainIdx] for m in matches]).reshape(-1, 1, 2)
        ransacReprojThre = 4
        #使用RANCAC选择最优的四组匹配点，再计算H矩阵
        H, status = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThre)

        #计算image1的mask
        mask = np.ones((image1.shape[0], image1.shape[1], 3), dtype="float")
        mask_hed = cv2.warpPerspective(mask, H, (image2.shape[1], image2.shape[0]))
        mask_hed = (255*mask_hed).astype(np.uint8)
        cv2.imwrite('./image/result/mask_hed.jpg', mask_hed)

        # 进行图像校正
        image_perspective = cv2.warpPerspective(image1, H, (image2.shape[1], image2.shape[0]))
        image_perspective = image_perspective.astype(np.uint8)
        cv2.imwrite('./image/result/image_per.jpg', image_perspective)
        image2=image2.astype(np.uint8)

        # 计算mask的中心点  使用外切矩形法求得几何中点
        cnts = cv2.findContours(cv2.cvtColor(mask_hed, cv2.COLOR_RGB2GRAY), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0]   
        x, y, w, h = cv2.boundingRect(cnts[0])
        cX = x + w//2
        cY = y + h//2
        center = (cX, cY)

        # 进行poisson融合
        result = cv2.seamlessClone(image_perspective, image2, mask_hed, center, cv2.NORMAL_CLONE)
        result = result/255
        return result, status
 ```
 
 ### 4.图像拼接与融合（Alpha Blending）
 ```
 def StitchAndBlend(image1, image2):
    kps1, des1 = FeatureExtract(image1)
    kps2, des2 = FeatureExtract(image2)
    matches = FeatureMatch(des1, des2)
    # 当筛选项的匹配对大于4对时：计算视角变换矩阵
    if len(matches) > 4:
        pts1 = np.float32([kps1[m.queryIdx] for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kps2[m.trainIdx] for m in matches]).reshape(-1, 1, 2)
        ransacReprojThre = 4
        #使用RANCAC选择最优的四组匹配点，再计算H矩阵
        H, status = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThre)
        #计算mask，并对其进行羽化
        mask = np.ones((image1.shape[0], image1.shape[1], 3), dtype="float")
        mask_hed = cv2.warpPerspective(mask, H, (image2.shape[1], image2.shape[0]))
        mask_ite = mask_hed.copy()
        for _ in range(80):
            mask_blured = cv2.blur(mask_ite, (40,40))
            mask_ite = mask_blured.copy()
        result1 = cv2.warpPerspective(image1, H, (image2.shape[1], image2.shape[0]))
        result1 = result1.astype(float)
        result1 = cv2.multiply(mask_blured, result1)
        cv_show(' ',result1/255)
        result2 = np.zeros((image2.shape[0], image2.shape[1], 3), np.uint8)
        result2[0:image2.shape[0], 0:image2.shape[1]] = image2
        result2 = result2.astype(float)
        result2 = cv2.multiply(1-mask_blured, result2)
        result = cv2.add(result1, result2)
        result = result/255
        cv_show('result', result)
        return result, status
 ```
结果展示
---
### 1.待嵌入图像

![](https://github.com/USTC-Computer-Vision-2021/project-cv-szy-lrx/blob/main/source/left1.jpg)
![](https://github.com/USTC-Computer-Vision-2021/project-cv-szy-lrx/blob/main/source/right3.png)
### 2.匹配图像（Poisson Blending）

![](https://github.com/USTC-Computer-Vision-2021/project-cv-szy-lrx/blob/main/result/matched_img.jpg)
### 3.结果图像（Poisson Blending）

![](https://github.com/USTC-Computer-Vision-2021/project-cv-szy-lrx/blob/main/result/result_poisson.jpg)

### 4.结果图像（Alpha Blending）

![](https://github.com/USTC-Computer-Vision-2021/project-cv-szy-lrx/blob/main/result/result_alpha.jpg)


## 工程说明

### 代码环境
- python >= 3.7
- opencv-python >= 4.5.2

### 工程结构
```
.
├── image
│   ├── ....# 带拼接的原始图片
├── README.md
├── result
│   ├── matched_img.jpg
│   ├── result_alpha.jpg
│   └── result_poisson.jpg
├── source  
│   ├── cv_alpha.py
│   └── cv_poisson.py
└── USED-Image
    ├── hw_1.png
    └── hw1.png
```
### 运行说明
- 执行 python ./source/cv_alpha.py 得到使用alpha融合的结果
- 执行 python ./source/cv_poisson.py 得到使用poisson融合的结果
- 更换图片时，只需修改相应.py文件中 image1、image2的路径即可



