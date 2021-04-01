import cv2

#调用摄像头资源，0代表电脑摄像头，1代表外接摄像头(usb摄像头)
cap = cv2.VideoCapture(0)
#设置显示宽度和高度,3:宽，4：高
cap.set(3, 900)
cap.set(4, 900)
#检测摄像头是否可以获取
while  cap.isOpened():
   #获取摄像头数据
    raval, frame = cap.read()
    #显示图像
    cv2.imshow('Capture', frame)
    k = cv2.waitKey(1)
 #点击s键继续
    if k == ord('s'):
        print(cap.get(3))
        print(cap.get(4))
  #点击q键退出
    elif k == ord('q'):
        print("调用结束")
        break

cap.release()
cv2.destoryAllWindows()