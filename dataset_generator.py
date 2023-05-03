'''

【此文件用于生成数据集。在推测时，也使用此文件进行图像分割】

使用segment anything的方法，对一个文件夹内的所有图片进行分割，将每个分块的图片保存到一个文件夹内
使用方法：python3 segment.py -i input_dir -o output_dir
input_dir:输入文件夹，包含所有待分割的图片
output_dir:输出文件夹，包含所有分割后的图片
注意：输入文件夹内的图片必须是png格式，输出文件夹内的图片也是png格式
注意：输入文件夹内的图片的文件名必须是数字，如1.png,2.png,3.png等
输出内容的命名规则：*-*.png，如1-1.png,1-2.png,2-1.png,2-2.png等
用SamAutomaticMaskGenerator生成的mask是一个列表，每个元素是一个字典，字典的keys有：
-segmentation : the mask
-area : the area of the mask in pixels
-bbox : the boundary box of the mask in XYWH format
-predicted_iou : the model's own prediction for the quality of the mask
-point_coords : the sampled input point that generated this mask
-stability_score : an additional measure of mask quality
-crop_box : the crop of the image used to generate this mask in XYWH format
使用masks前需要把它从pycocotools的COCO格式转换为numpy格式
'''
#载入必要的库
from segment_anything import SamAutomaticMaskGenerator,sam_model_registry
import cv2
import argparse
import numpy as np
import sys
import os
import pycocotools.mask as mask_utils
import time
sys.path.append("..")

def to_numpy(masks):
    """Converts masks from COCO format to numpy arrays.
    Args:
        masks: A list of dicts in COCO mask format.
    Returns:
        A list of numpy arrays.
    """
    return [m['segmentation'] for m in masks]
#载入模型（控制台输出加载模型用时）
print('Loading model...')
start = time.time()
sam = sam_model_registry['vit_h'](checkpoint="sam_vit_h_4b8939.pth")
sam.to(device='cuda')
end = time.time()
print('Done! Time cost: ' + str(end - start) + 's')
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=16,
    pred_iou_thresh=0.9,
    stability_score_thresh=0.92,
    crop_n_layers=0,
    crop_nms_thresh=0.9,
    crop_overlap_ratio=0.1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=500,  # Requires open-cv to run post-processing
)
#通过命令行参数获取输入输出文件夹
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dir', type=str, default='original_data')
parser.add_argument('-o', '--output_dir', type=str, default='dataset')
args = parser.parse_args()
input_dir = args.input_dir
output_dir = args.output_dir
#遍历输入文件夹内的所有图片,i是图片的文件名
for i in os.listdir(input_dir):
    #读取图片
    print('processing image ' + i + '  ...  ')
    img = cv2.imread(input_dir + '/' + i)
    #分割图片
    masks = mask_generator.generate(img)
    print('Seg √ ')
    #将masks转换为受支持的numpy格式
    masks = to_numpy(masks)
    print('ToNumpy √ ')
    #将分割后的图片保存到输出文件夹内
    for j in range(len(masks)):
        # 获取当前掩码
        mask = masks[j]
        # 将掩码转换为与原图相同的数据类型
        mask = mask.astype(img.dtype)
        # 将原图和掩码进行按位与运算
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        # 找到掩码的轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 计算轮廓的边界框
        x, y, w, h = cv2.boundingRect(contours[0])
        # 裁剪图像
        cropped_img = masked_img[y:y+h, x:x+w]
        # 保存图像
        cv2.imwrite(output_dir + '/' + i[:-4] + '-' + str(j) + '.png', cropped_img)
    print('Done!')
