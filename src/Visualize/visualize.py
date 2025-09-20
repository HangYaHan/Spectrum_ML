import cv2
import os

def draw_rois_on_image(img_path, rois_path, save_path=None):
	"""
	读取 rois.txt 并将所有 ROI 画在图片上，保存或显示。
	img_path: 原始图片路径
	rois_path: ROI坐标文件路径
	save_path: 可选，保存可视化图片路径
	"""
	img = cv2.imread(img_path)
	if img is None:
		print(f"Failed to load image: {img_path}")
		return
	rois = []
	with open(rois_path, 'r') as f:
		for line in f:
			parts = line.strip().split(',')
			if len(parts) == 4:
				x, y, w, h = map(int, parts)
				rois.append((x, y, w, h))
	# 画框
	for i, (x, y, w, h) in enumerate(rois, 1):
		cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
		cv2.putText(img, str(i), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
	# 保存或显示
	if save_path:
		cv2.imwrite(save_path, img)
	else:
		cv2.imshow('ROIs on Image', img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

# 示例调用
if __name__ == "__main__":
	img_path = os.path.join('RawData', 'pics', '0001.jpg')
	rois_path = os.path.join('ProcessedData', 'FinalCSV', 'rois.txt')
	save_path = os.path.join('ProcessedData', 'FinalCSV', '0001_with_rois.jpg')
	draw_rois_on_image(img_path, rois_path, save_path)