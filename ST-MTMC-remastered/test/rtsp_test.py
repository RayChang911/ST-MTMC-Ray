import cv2

if __name__ == '__main__':

    # 開啟 RTSP 串流
    vidCap = cv2.VideoCapture('rtsp://localhost/test')

    # 建立視窗
    cv2.namedWindow('image_display', cv2.WINDOW_AUTOSIZE)

    while True:
        # 從 RTSP 串流讀取一張影像
        ret, image = vidCap.read()

        if ret:
            # 顯示影像
            cv2.imshow('image_display', image)
            cv2.waitKey(10)
        else:
            # 若沒有影像跳出迴圈
            break

    # 釋放資源
    vidCap.release()

    # 關閉所有 OpenCV 視窗
    cv2.destroyAllWindows()