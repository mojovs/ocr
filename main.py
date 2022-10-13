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
#cv_show(tophat,'gradx')
#进行梯度运算
gradx = cv2.Sobel(tophat,cv2.CV_32F,dx=1,dy=0,ksize=-1) #3x3的kernel
#cv_show(gradx,'gradx')

gradx = np.absolute(gradx)
(minVal,maxVal) = (np.min(gradx),np.max(gradx)) #提取出最大最小值

gradx = (255*((gradx-minVal)/(maxVal-minVal)))
gradx = gradx.astype(np.uint8)
print("gradx.shape:",np.array(gradx.shape))
#cv_show(gradx,'gradx')

#开始进行闭操作
gradx = cv2.morphologyEx(gradx,cv2.MORPH_CLOSE,rectKernel)
#cv_show(gradx,'gradx')
#自动阈值处理,把模糊的图像清晰
thresh = cv2.threshold(gradx,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
#cv_show(thresh,'thresh')

#再来一个闭操作
thresh= cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,sqKernel)
#cv_show(thresh,'thresh')

# 给输入图片画上轮廓
threshCnts,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cnts = threshCnts
cur_img = image.copy()
cv2.drawContours(cur_img,cnts,-1,(0,0,255),3)
cv_show(cur_img,'cur_image')

locs = []
# 遍历轮廓
for (i,c) in enumerate(cnts):
    # 计算矩形
    (x,y,w,h) = cv2.boundingRect(c)
    ar = w/float(h) # 获取比例

    # 选择合适的区域
    if ar>2.5 and ar <4.0:
        if(w>40 and w<55) and (h>10 and h <20):
            locs.append((x,y,w,h))
# 对获取到卡号组轮廓进行排序
locs = sorted(locs,key=lambda x:x[0])

# 对每个卡号组轮廓中的数字进行遍历
output =[]
for (i,(gx,gy,gw,gh)) in enumerate(locs):
    groupOutput = []
    # 根据坐标提取出一个灰度图组
    group= gray[gy-5:gy+gh+5,gx-5:gx+gw+5]

    #对每一个组进行二值化处理
    group = cv2.threshold(group,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
    cv_show(group,'group')

    #对每个组里进行再取轮廓
    inside_cnts,_= cv2.findContours(group,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);
    #画轮廓
    # 对轮廓进行排序
    inside_cnts = myutils.sort_contours(inside_cnts,method="left-to-right")[0]
    #画一下轮廓看下
    inside_group_contours = group.copy()
    cv2.drawContours(inside_group_contours,inside_cnts, -1, (0, 0, 255), 3)

    # 再对这些轮廓进行分别绘制
    for (i,pnt_set) in enumerate(inside_cnts):
        (x,y,w,h)= cv2.boundingRect(pnt_set)
        roi = group[y:y+h,x:x+w] #获取到每个数字图片
        # 对获取到的图像进行resize,从而容易匹配模板
        roi = cv2.resize(roi, (57, 88))
        #cv_show(roi,"roi")
        scores= []

        # 模板匹配 针对每个数，他与模板图片的匹配得分
        for(digit,digitROI) in digits.items():
            # 模板匹配
            result = cv2.matchTemplate(roi,digitROI,cv2.TM_CCOEFF)
            (_,score,_,_) = cv2.minMaxLoc(result)
            scores.append(score)

        # 得到适合的数字
        groupOutput.append(str(np.argmax(scores))) #寻找最大得分值的索引

    # 画图
    cv2.rectangle(image,(gx-5,gy-5) ,(gx+gw+5,gy+gh+5),(0,0,255),1)
    cv2.putText(image,"".join(groupOutput),(gx,gy-15),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,255),2)

    output.extend(groupOutput)



cv_show(image,"image")








