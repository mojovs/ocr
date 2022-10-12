import cv2
import argparse
import numpy as np
import myutils

# 分析传参
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="path to input image")
ap.add_argument("-t","--template",required=True,help="path to ocr template image")
args = vars(ap.parse_args())

first_number = {
    "3":"USA Express",
    "4":"Visa",
    "5":"MasterCard",
    "6":"Discover Card"
}

def cv_show(img,window):
    '''
    :param img: 输入的图片
    :param window:显示窗口名称
    :return:
    '''
    cv2.imshow(window,img)
    cv2.waitKey(0)
    cv2.destroyWindow(window)

#先导入模板图片
img = cv2.imread(args["template"])

# 模板转换 成灰度图
ref = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #模板灰度图
# 对模板进行二值化反转
ret ,ref = cv2.threshold(ref,10, 255, cv2.THRESH_BINARY_INV)
#cv_show(ref,'ref')

# 计算轮廓
refCnts,_=cv2.findContours(ref.astype(np.uint8).copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #轮廓点

# 画出轮廓
cv2.drawContours(img,refCnts,-1,(0,0,255),3)
#cv_show(img,'img')
# 打印下refCnts
print(np.array(refCnts).shape)
refCnts = myutils.sort_contours(refCnts,method="left-to-right")[0] #给轮廓排下序 按照x坐标排序

digits={}

#对轮廓进行遍历
for (i,c) in enumerate(refCnts):
    # 计算外接矩形
    (x,y,w,h) = cv2.boundingRect(c)
    # 获取 轮廓图像的大小uu
    roi = ref[y:y+h,x:x+w]
    # 对获取到的图像进行resize
    roi = cv2.resize(roi,(57,88))
    # 将图片存入模板
    digits[i]=roi

#制作卷积核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

# 读入要识别的图像 ，预处理
image=cv2.imread(args['image'])
cv_show(image,'image')
image = myutils.resize(image, width=300) #按比例缩放

#进行灰度处理
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#先顶帽
tophat = cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,rectKernel)
#进行梯度运算
gradx = cv2.Sobel(tophat,cv2.CV_32F,dx=1,dy=0,ksize=-1) #3x3的kernel
gradx = np.absolute(gradx)
(minVal,maxVal) = (np.min(gradx),np.max(gradx))

gradx = (255*((gradx-minVal)/(maxVal-minVal)))
gradx = gradx.astype(np.uint8)

print(np.array(gradx.shape))
cv_show(gradx,'gradx')