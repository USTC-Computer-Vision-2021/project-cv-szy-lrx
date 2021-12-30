import cv2
import numpy as np
from  PIL import Image

#显示图片
def cv_show(name, result):
    cv2.imshow(name, result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#使用SIFT算法进行特征提取，返回特征点坐标及特征描述
def FeatureExtract(image):
    descriptor = cv2.SIFT_create()
    kps, des = descriptor.detectAndCompute(image, None)
    #将返回的特征点转化为numpy格式的数组
    kps = np.float32([kp.pt for kp in kps])
    return kps, des

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

#画出两图关键点的匹配关系
def DrawMatches(image1, image2, kps1, kps2, matches, status):
    (h1, w1) = image1.shape[:2]
    (h2, w2) = image2.shape[:2]
    result = np.zeros((max(h1, h2), w1 + w2, 3), dtype="uint8")
    result[0:h1, 0:w1] = image1
    result[0:h2, w1:] = image2
    matchnp = []
    for m in matches:
        matchnp.append((m.trainIdx, m.queryIdx))
    for ((trainIdx, queryIdx), s) in zip(matchnp, status):
        # 当点对匹配成功时，画到结果图中
        if s == 1:
            # 画出匹配对
            pt1 = (int(kps1[queryIdx][0]), int(kps1[queryIdx][1]))
            pt2 = (int(kps2[trainIdx][0]) + w1, int(kps2[trainIdx][1]))
            cv2.line(result, pt1, pt2, (0, 255, 0), 1)
    return result

#图像拼接与融合
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

image1 = cv2.imread('./image/right3.png')
cv_show("image1", image1)
image2 = cv2.imread('./image/left1.jpg')
cv_show("image2", image2)
(kps1, features1) = FeatureExtract(image1)
(kps2, features2) = FeatureExtract(image2)
matches = FeatureMatch(features1, features2)
result, status = StitchAndBlend(image1, image2)
matchedimg = DrawMatches(image1, image2, kps1, kps2, matches, status)
cv_show('match',matchedimg)
cv2.imwrite('./result/result_poisson.jpg', np.array(result))