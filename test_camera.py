import cv2

if __name__ == "__main__":
    # 打开摄像头 1
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("无法打开摄像头 1")
        exit()

    print("按回车键保存截图，按 ESC 键退出程序。")

    while True:
        # 读取摄像头画面
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break

        # 显示画面
        cv2.imshow("Camera 1", frame)

        # 检测按键
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC 键退出
            break
        elif key == 13:  # 回车键保存截图
            cv2.imwrite("screenshot.png", frame)
            print("截图已保存为 screenshot.png")

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
