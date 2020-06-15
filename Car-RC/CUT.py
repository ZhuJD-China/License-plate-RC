import os

import cv2

# img1=cv2.imread('@d4405b26-fcfc-4c5b-bd7a-84bb49d79e93.jpg', 0)
# print(img1.shape)
# for i in img1:
#     print(i)
# cv2.imshow("1",img1)
# cv2.imshow("2",img2)
# cv2.imwrite("7.png",img1)

# https://www.cnblogs.com/do-hardworking/p/9829151.html

path = "./car/"


def threshold_demo(image):
    ret, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    print("threshold value: %s" % ret)  # 打印阈值，前面先进行了灰度处理0-255，我们使用该阈值进行处理，低于该阈值的图像部分全为黑，高于该阈值则为白色
    cv2.imshow("binary", binary)  # 显示二值化图像
    return binary


def threshold_demo_2(image):
    retval, dst = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)

    # # 腐蚀和膨胀是对白色部分而言的，膨胀，白区域变大，最后的参数为迭代次数
    # dst = cv2.dilate(dst, None, iterations=1)
    # # 腐蚀，白区域变小
    # dst = cv2.erode(dst, None, iterations=4)
    # cv2.imshow("binary2", dst)
    return dst


def threshold_demo_3(image):
    img_thre = image
    cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV, img_thre)
    cv2.imshow("binary3", img_thre)
    cv2.waitKey(0)
    return img_thre


def find_end(start, arg, black, white, width, black_max, white_max):
    end = start + 1
    for m in range(start + 1, width - 1):
        if (black[m] if arg else white[m]) > (0.98 * black_max if arg else 0.98 * white_max):
            end = m
            break
    return end


# img_thre=threshold_demo_2(img1)
# thresh=img_thre

def accss_pixels(image):
    height, width = image.shape
    for row in range(height):
        for list in range(width):
                pv = image[row, list]
                image[row, list] = 255 -pv
    return image


def split(thresh, name):
    thresh = accss_pixels(thresh)
    # print(thresh)
    # 新建文件夹
    os.mkdir(name)
    # 写入二极化的车牌号
    url = name + '/car_LC.bmp'
    print(url)
    # print(thresh)
    cv2.imwrite(url, thresh)
    # 分割字符
    '''
    判断底色和字色 
    '''
    # 记录黑白像素总和
    white = []
    black = []
    height = thresh.shape[0]  # 263
    width = thresh.shape[1]  # 400
    # print('height',height)
    # print('width',width)
    white_max = 0
    black_max = 0
    # 计算每一列的黑白像素总和
    for i in range(width):
        line_white = 0
        line_black = 0
        for j in range(height):
            if thresh[j][i] == 255:
                line_white += 1
            if thresh[j][i] == 0:
                line_black += 1
        white_max = max(white_max, line_white)
        black_max = max(black_max, line_black)
        white.append(line_white)
        black.append(line_black)
    # arg为true表示黑底白字，False为白底黑字
    arg = True
    if black_max < white_max:
        arg = False

    n = 1
    start = 1
    end = 2
    i = 1
    while n < width - 2:
        n += 1
        # 判断是白底黑字还是黑底白字  0.05参数对应上面的0.95 可作调整
        if (white[n] if arg else black[n]) > (0.02 * white_max if arg else 0.02 * black_max):
            start = n
            end = find_end(start, arg, black, white, width, black_max, white_max)
            n = end
            if end - start > 5:
                cj = thresh[1:height, start:end]
                cv2.imwrite(name + '/' + str(i) + '.png', cj)
                i += 1
                # cv2.imshow('cutlicense'+str(i), cj)
                # cv2.waitKey(0)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def opt(name):
    img1 = cv2.imread(path + name + '.jpg', 0)
    # print(img1)
    # im = cv2.imdecode(np.fromfile(r'C:\Users\83815\Desktop\1\1.jpg', dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    # img1=cv2.imdecode(np.fromfile(path+'\\'+name+'.jpg', dtype=np.uint8),0)
    # cv2.imwrite('test.png',img1)
    split(threshold_demo_2(img1), name)


def main():
    files = os.listdir(path)
    print(type(files[0]))
    print(len(files))
    for i in files:
        opt(i[:-4])
        # print(i[:-4])
        # print(type(i[:-4]))

    # ord(str)>19968 and ord(str)<40869
    # for i in files:
    #     if ord(i[0])>19968 and ord(i[0])<40869:
    #         old_name=path+i
    #         new_name=path+i[1:]
    #         os.rename(old_name,new_name)
    #         print("修改成功")


try:
    main()
except cv2.error:
    pass
