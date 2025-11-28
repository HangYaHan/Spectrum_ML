import cv2
import os
from datetime import datetime


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"无法创建目录 {path}: {e}")


if __name__ == "__main__":
    # 打开摄像头 1
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开摄像头 1")
        exit()

    screenshots_dir = "screenshots"
    ensure_dir(screenshots_dir)

    print("按回车键保存截图，按 ESC 键退出程序。每次保存会生成一个唯一文件名（不会覆盖）。")

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
        elif key in (13, 10):  # 回车键保存截图 (CR and LF)
            # 生成基于时间戳的唯一文件名，保留毫秒
            ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            filename = f"screenshot_{ts}.png"
            filepath = os.path.join(screenshots_dir, filename)
            try:
                cv2.imwrite(filepath, frame)
                print(f"截图已保存为 {filepath}")
            except Exception as e:
                print(f"保存截图失败: {e}")

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
