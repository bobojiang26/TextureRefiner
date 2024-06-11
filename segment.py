import cv2
from PIL import Image
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

# 加载图像
image_path = 'back.png'
image = cv2.imread(image_path)

# 使用 Segment Anything 模型
sam = sam_model_registry["default"](checkpoint="/home/zcb/self_code_training/InTeX_self/sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)

# 准备图像
image = Image.open(image_path).convert("RGB")
image = np.array(image)

# 进行分割
predictor.set_image(image)
masks, scores, _ = predictor.predict()

# 对分割结果进行处理和标注
for mask, score in zip(masks, scores):
    # 这里可以根据mask和score进行进一步处理
    # 示例中简单展示mask和score
    print("Mask:", mask)
    print("Score:", score)

# 在图像上绘制标注
for idx, mask in enumerate(masks):
    label = f"Part {idx+1}"
    # 获取掩码的边界框
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8) * 255)
    # 绘制边界框和标签
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# 显示或保存标注后的图像
output_path = 'output_labeled_image.jpg'
cv2.imwrite(output_path, image)