import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

CADES_PATH = 'haarcascade_frontalface_alt2.xml'  # 创建人脸检测器


def face_detect(img_path):  # 人脸检测
    color = (0, 255, 0)
    img_bgr = cv2.imread(img_path)
    classifier = cv2.CascadeClassifier(CADES_PATH)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    facerects = classifier.detectMultiScale(img_gray)
    if len(facerects) > 0:
        for rect in facerects:
            x, y, w, h = rect
            if w > 200:
                cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, 2)
    # cv2.imshow('detect', img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('faceDetect.png', img_bgr)


def my_guidedFilter_oneChannel(srcImg, guidedImg, rad=9, eps=0.01):
    srcImg = srcImg / 255.0
    guidedImg = guidedImg / 255.0
    img_shape = np.shape(srcImg)

    P_mean = cv2.boxFilter(srcImg, -1, (rad, rad), normalize=True)
    I_mean = cv2.boxFilter(guidedImg, -1, (rad, rad), normalize=True)

    I_square_mean = cv2.boxFilter(np.multiply(guidedImg, guidedImg), -1, (rad, rad), normalize=True)
    I_mul_P_mean = cv2.boxFilter(np.multiply(srcImg, guidedImg), -1, (rad, rad), normalize=True)

    var_I = I_square_mean - np.multiply(I_mean, I_mean)
    cov_I_P = I_mul_P_mean - np.multiply(I_mean, P_mean)

    a = cov_I_P / (var_I + eps)
    b = P_mean - np.multiply(a, I_mean)

    a_mean = cv2.boxFilter(a, -1, (rad, rad), normalize=True)
    b_mean = cv2.boxFilter(b, -1, (rad, rad), normalize=True)

    dstImg = np.multiply(a_mean, guidedImg) + b_mean

    return dstImg * 255.0


def my_guidedFilter_threeChannel(srcImg, guidedImg, rad=9, eps=0.01):
    img_shape = np.shape(srcImg)

    dstImg = np.zeros(img_shape, dtype=float)

    for ind in range(0, img_shape[2]):
        dstImg[:, :, ind] = my_guidedFilter_oneChannel(srcImg[:, :, ind],
                                                       guidedImg[:, :, ind], rad, eps)

    dstImg = dstImg.astype(np.uint8)

    return dstImg


def psnr(A, B):
    val = 255
    mse = ((A.astype(np.float) - B) ** 2).mean()
    return 10 * np.log10((val * val) / mse)


def double2unit8(I, L, ratio=1.0, sigma=20.0):
    I = I.astype(np.float64)  # 转换成float形式便于计算
    noise = np.random.randn(*I.shape) * sigma  # 生成的时形状与I相同，均值为0，标准差为sigam的随机高斯噪声矩阵
    noisy = I + noise  # 噪声图像
    return np.clip(np.round(noisy * ratio), 0, 255).astype(L.dtype)  # 转换回图像的unit8均值


def make_kernel(f):  # 计算得到一个高斯核，用于后续的计算
    kernel = np.zeros((2 * f + 1, 2 * f + 1))
    for d in range(1, f + 1):
        kernel[f - d:f + d + 1, f - d:f + d + 1] += (1.0 / ((2 * d + 1) ** 2))
    return kernel / kernel.sum()


def NLmeansfilter(I, L, h_=10, templateWindowSize=5, searchWindowSize=11):  # NLM滤波
    I = I.astype(np.float64)
    f = int(templateWindowSize / 2)
    t = int(searchWindowSize / 2)
    height, width = I.shape[:2]  # 利用ndarray的索引得到长宽
    padLength = t + f
    I2 = np.pad(I, padLength, 'symmetric')  #
    kernel = make_kernel(f)
    h = (h_ ** 2)
    I_ = I2[padLength - f:padLength + f + height, padLength - f:padLength + f + width]

    average = np.zeros(I.shape)
    sweight = np.zeros(I.shape)
    wmax = np.zeros(I.shape)
    for i in range(-t, t + 1):
        for j in range(-t, t + 1):
            if i == 0 and j == 0:
                continue
        I2_ = I2[padLength + i - f:padLength + i + f + height, padLength + j - f:padLength + j + f + width]
        w = np.exp(-cv2.filter2D((I2_ - I_) ** 2, -1, kernel) / h)[f:f + height, f:f + width]
        sweight += w
        wmax = np.maximum(wmax, w)
        average += (w * I2_[f:f + height, f:f + width])
    I1 = (average + wmax * I) / (sweight + wmax)
    return np.clip(np.round(I1), 0, 255).astype(L.dtype)


def fangcha(img):  # 方差函数
    row = img.shape[0]
    col = img.shape[1]
    varImg = np.zeros([row, col])
    for i in range(row):  # 求取方差范围
        for j in range(col):
            if i - 5 > 0:
                up = i - 5
            else:
                up = 0
            if i + 5 < row:
                down = i + 5
            else:
                down = row
            if j - 5 > 0:
                left = j - 5
            else:
                left = 0
            if j + 5 < col:
                right = j + 5
            else:
                right = col
            window = img[up:down, left:right]
            mean, var = cv.meanStdDev(window)  # 调用OpenCV函数求取均值和方差
            varImg[i, j] = var
    return varImg


def qiuquan(img1, img2):  # 求权值
    row = img1.shape[0]
    col = img1.shape[1]
    array1 = fangcha(img1)  # 调用求方差函数
    array2 = fangcha(img2)
    for i in range(row):  # 求权
        for j in range(col):
            weight1 = array1[i, j] / (array1[i, j] + array2[i, j])
            weight2 = array2[i, j] / (array1[i, j] + array2[i, j])
            array1[i, j] = weight1
            array2[i, j] = weight2
    return array1, array2


def ronghe(img1, img2):  # 图像融合
    cc = img1.copy()
    b, g, r = cv.split(img1)  # 分通道处理
    b1, g1, r1 = cv.split(img2)
    weight1, weight2 = qiuquan(b, b1)  # 调用求权重函数
    weight11, weight22 = qiuquan(g, g1)
    weight111, weight222 = qiuquan(r, r1)
    new_img = img1 * 1
    row = new_img.shape[0]
    col = new_img.shape[1]
    b2, g2, r2 = cv.split(cc)
    for i in range(row):  # 图像融合
        for j in range(col):
            b2[i, j] = (weight1[i, j] * b[i, j] + weight2[i, j] * b1[i, j]).astype(int)
            g2[i, j] = (weight11[i, j] * g[i, j] + weight22[i, j] * g1[i, j]).astype(int)
            r2[i, j] = (weight111[i, j] * r[i, j] + weight222[i, j] * r1[i, j]).astype(int)
    new_img = cv.merge([b2, g2, r2])  # 通道合并
    return new_img


def gray2rgb(rgb, imggray):
    # 原图 R G 通道不变，B 转换回彩图格式
    R = rgb[:, :, 0]
    G = rgb[:, :, 1]
    B = ((imggray) - 0.299 * R - 0.587 * G) / 0.114

    grayRgb = np.zeros((rgb.shape))
    grayRgb[:, :, 2] = B
    grayRgb[:, :, 0] = R
    grayRgb[:, :, 1] = G
    return grayRgb


# Grayscale
def BGR2GRAY(img):
    # Grayscale
    gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
    return gray


# Gabor Filter
def Gabor_filter(K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0):
    # get half size
    d = K_size // 2
    # prepare kernel
    gabor = np.zeros((K_size, K_size), dtype=np.float32)
    # each value
    for y in range(K_size):
        for x in range(K_size):
            # distance from center
            px = x - d
            py = y - d
            # degree -> radian
            theta = angle / 180. * np.pi
            # get kernel x
            _x = np.cos(theta) * px + np.sin(theta) * py
            # get kernel y
            _y = -np.sin(theta) * px + np.cos(theta) * py
            # fill kernel
            gabor[y, x] = np.exp(-(_x ** 2 + Gamma ** 2 * _y ** 2) / (2 * Sigma ** 2)) * np.cos(
                2 * np.pi * _x / Lambda + Psi)
    # kernel normalization
    gabor /= np.sum(np.abs(gabor))
    return gabor


# 使用Gabor滤波器作用于图像上
def Gabor_filtering(gray, K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0):
    # get shape
    H, W = gray.shape
    # padding
    gray = np.pad(gray, (K_size // 2, K_size // 2), 'edge')
    # prepare out image
    out = np.zeros((H, W), dtype=np.float32)
    # get gabor filter
    gabor = Gabor_filter(K_size=K_size, Sigma=Sigma, Gamma=Gamma, Lambda=Lambda, Psi=0, angle=angle)
    # filtering
    for y in range(H):
        for x in range(W):
            out[y, x] = np.sum(gray[y: y + K_size, x: x + K_size] * gabor)
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)
    return out


# 使用6个不同角度的Gabor滤波器对图像进行特征提取
def Gabor_process(img):
    # get shape
    H, W = img.shape
    As = [0, 30, 60, 90, 120, 150]
    # prepare pyplot
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0.2)
    out = np.zeros([H, W], dtype=np.float32)
    # each angle
    for i, A in enumerate(As):
        # gabor filtering
        _out = Gabor_filtering(img, K_size=11, Sigma=1.5, Gamma=1.2, Lambda=3, angle=A)
        # add gabor filtered image
        out += _out
    # scale normalization
    out = out / out.max() * 255
    out = out.astype(np.uint8)
    return out


def main():
    input_png = input('请输入"图片名.png"（例如:test.png）：')
    img = cv2.imread(input_png)
    # print(np.shape(img))
    dstimg = my_guidedFilter_threeChannel(img, img, 9, 0.01)
    # print(np.shape(dstimg))

    # 人脸检测
    face_detect(input_png)

    # 引导滤波
    cv2.imwrite('guidedFilter.png', dstimg)

    # NLM滤波
    nlmimg = cv2.fastNlMeansDenoisingColored(img, None, 20, 5, 11)
    cv2.imwrite('nlmFilter.png', nlmimg)

    # 皮肤检测

    img5 = dstimg

    px = img5[150, 200]

    blue = img5[150, 200, 0]

    green = img5[150, 200, 1]

    red = img5[150, 200, 2]
    img5[150, 200] = [0, 0, 0]

    blue = img5.item(100, 200, 0)

    green = img5.item(100, 200, 1)

    red = img5.item(100, 200, 2)
    img5.itemset((100, 200, 1), 255)

    green = img5.item(100, 200, 1)
    rows, cols, channels = img5.shape

    imgSkin = np.zeros(img5.shape, np.uint8)
    imgSkin = img5.copy()

    for r in range(rows):
        for c in range(cols):
            # get pixel value
            B = img5.item(r, c, 0)
            G = img5.item(r, c, 1)
            R = img5.item(r, c, 2)
            skin = 0
            if (abs(R - G) > 15) and (R > G) and (R > B):
                if (R > 95) and (G > 40) and (B > 20) and (max(R, G, B) - min(R, G, B) > 15):
                    skin = 1
                elif (R > 220) and (G > 210) and (B > 170):
                    skin = 1

            if 0 == skin:
                imgSkin.itemset((r, c, 0), 0)
                imgSkin.itemset((r, c, 1), 0)
                imgSkin.itemset((r, c, 2), 0)

    img5 = cv2.cvtColor(img5, cv2.COLOR_BGR2RGB)
    imgSkin = cv2.cvtColor(imgSkin, cv2.COLOR_BGR2RGB)

    cv2.imwrite('skinDetect.png', imgSkin)

    # 融合
    img1 = ronghe(img, dstimg)
    cv.imwrite('imgFusion.png', img1)  #

    # 锐化
    img2 = cv2.imread('guidedFilter.png', 0)
    # img3 = cv2.imread('g1-nlm.png', 1)
    out = Gabor_process(img2)
    cv2.imwrite('gabor.png', out)
    img3 = cv2.imread('gabor.png')
    img4 = cv2.add(dstimg, img3)
    cv2.imshow('finalResult', img4)
    cv2.imwrite('finalResult.png', img4)
    # print('美颜成功！')

    cv2.waitKey(0)


if __name__ == '__main__':
    main()
