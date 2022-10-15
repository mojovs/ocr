import cv2
import argparse
import numpy as np
import myutils

# 分析传参
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="path to input image")
args = vars(ap.parse_args())
def cv_show(img,window):
    '''
    :param img: 输入的图片
    :param window:显示窗口名称
    :return:
    '''
    cv2.imshow(window,img)
    cv2.waitKey(0)
    cv2.destroyWindow(window)

def order_points(pts):
    '''
    对4个轮廓点进行排序
    :param pts: 轮廓点集
    :return: rect 排序好的4点
    '''

    rect = np.zeros((4,2),dtype='float32')
    #按照顺序找到对应的坐标
    s = pts.sum(axis=1) #x+y值
    rect[0] = pts[np.argmin(s)] #左上角
    rect[2] = pts[np.argmax(s)] #右下角

    #计算右上和左下
    diff = np.diff(pts,axis=1) #沿x轴相减
    rect[1] = pts[np.argmin(diff)] #右上角
    rect[3] = pts[np.argmax(diff)] #左下角
    return rect


def transform_4pnt(image,pnts):
    '''
    4点轮廓进行透视变换
    :param image: 输入图像
    :param pnts: 轮廓点
    :return: 一个矩形
    '''
    rect = order_points(pnts)
    (tl,tr,br,bl) = rect #提取出各个点位

    # 计算w和h
    widthA = np.sqrt((br[0]-bl[0])**2 + (br[1]-bl[0])**2) #勾股定理
    widthB = np.sqrt((tr[0]-tl[0])**2 + (tr[1]-tl[0])**2) #勾股定理
    heightA = np.sqrt((bl[0]-tl[0])**2 + (bl[1]-tl[0])**2) #勾股定理
    heightB = np.sqrt((br[0]-tr[0])**2 + (br[1]-tr[0])**2) #勾股定理
    #取最大长宽作为现在的长宽
    maxWidth= max(int(widthA),int(widthB))
    maxHeight = max(int(heightA),int(heightB))
    #变换对应
    dst =np.array([
        [0,0],
        [maxWidth-1,0],
        [maxWidth-1,maxHeight],
        [0,maxHeight-1],
    ],dtype='float32')

    # 计算变换矩阵
    M=cv2.getPerspectiveTransform(rect,dst)
    warped = cv2.warpPerspective(image,M,(maxWidth,maxHeight))
    return  warped




def main():
    #导入原始图片
    origin =  cv2.imread(args['image'])

    #设置图片大小
    ratio = origin.shape[0]/500.0
    origin_resize = myutils.resize(origin,height=500)

    # 处理边缘
    gray = cv2.cvtColor(origin_resize,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0) #去掉一些噪音
    edged = cv2.Canny(gray,75,200)
    cv_show(edged, 'edged')

    #需要最外面的轮廓
    contours = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    cnts = sorted(contours,key=cv2.contourArea,reverse=True)[:5] #取最大的5个轮廓

    for c in cnts:
        peri =cv2.arcLength(c,True) #计算轮廓周长
        # 0.02*peri是精度控制，epsilon越大，形状越归整，越小精度越高
        approx =cv2.approxPolyDP(c,0.02*peri,True) #计算近似
        #如果检测到的轮廓是4个点
        if len(approx) ==4:
            screenCnt = approx
            break

    #画出最外味的轮廓
    cv2.drawContours(origin_resize,[screenCnt] ,-1,(0,0,255),2)
    cv_show(origin_resize,'outline')

    #进行透视变换
    warped = transform_4pnt(origin,screenCnt.reshape(4,2)*ratio)

    #二值处理
    warped = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
    ref = cv2.threshold(warped,100,255,cv2.THRESH_BINARY)[1]
    cv_show(ref,'show')
    cv2.imwrite('scan.jpg',ref)

    #缩小下图片
    cv_show(myutils.resize(ref,height=600),'final')

if __name__ == "__main__":
    main()



