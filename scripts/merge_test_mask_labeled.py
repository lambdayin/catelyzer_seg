import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import glob
from typing import List, Tuple, Dict

import torch
from utils.data_loading import SelfDataset
from unet import UNet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help="Checkpoint .pth file for unet.")
    parser.add_argument('--input-dir', default='./test_0527/imgdata', type=str, help="Input directory containing images.")
    parser.add_argument('--output-dir', default='./test_0527/res', type=str, help="Output directory for results.")
    parser.add_argument('--grid-len', default=36, type=float, help="The physical length of each grid(mm).")
    parser.add_argument('--grid-pixel', default=651, type=float, help="Pixel distance of each grid.")
    parser.add_argument('--image-exts', default=['jpg', 'jpeg', 'png', 'bmp', 'tiff'], nargs='+', 
                       help="Image file extensions to process.")
    return parser.parse_args()


def get_image_files(input_dir, extensions):
    image_files = []
    for ext in extensions:
        pattern = os.path.join(input_dir, f"*.{ext}")
        image_files.extend(glob.glob(pattern, recursive=False))
        pattern = os.path.join(input_dir, f"*.{ext.upper()}")
        image_files.extend(glob.glob(pattern, recursive=False))
    return sorted(list(set(image_files)))


class CatalystDefectDetector:
    def __init__(self, len_per_pixel, config = None):
        """
        初始化检测器
        
        Args:
            config: 配置参数字典
        """
        self.len_per_pixel = len_per_pixel
        self.config = config or self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """获取默认配置参数"""
        return {
            # 轮廓检测参数
            'min_contour_area': 50,
            'max_contour_area': 10000,
            
            # 形状分析参数
            'aspect_ratio_threshold': 2.5,  # 长宽比阈值，用于区分条状和非条状
            'circularity_threshold': 0.3,   # 圆形度阈值
            'solidity_threshold': 0.85,      # 凸度阈值
            
            # 表面缺陷检测参数
            'roughness_threshold': 0.15,     # 粗糙度阈值
            'edge_variance_threshold': 50,   # 边缘方差阈值
            
            # 大小分析参数
            'size_outlier_factor': 3.0,      # 大小异常因子
            
            # 聚类参数
            'dbscan_eps': 0.3,
            'dbscan_min_samples': 3
        }
    
    def pre_process_single_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        mask_filter = yuv[:, :, 0] < 15
        mask_filter = mask_filter.astype(np.uint8)
        mask_eroded = cv2.dilate(mask_filter, np.ones((5, 5), np.uint8), iterations=2)
        mask_eroded = cv2.erode(mask_eroded, np.ones((5, 5), np.uint8), iterations=2)
            
        image[mask_eroded == 1] = [255, 255, 255]
        return image, mask_eroded 


    def inference_unet_batch(self, net, device, image_path):
        scale_factor = 0.5
        img_ori = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img_ori is None:
            raise ValueError(f"Cannot read image: {image_path}")
        net.eval()
        img = torch.from_numpy(SelfDataset.preprocess(None, img_ori, scale_factor, is_mask=False))    
        img = img.unsqueeze(0)           
        img = img.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            output = net(img).cpu()        
            mask = output.argmax(dim=1)
            output = (mask[0] * 255).squeeze().numpy().astype(np.uint8)
        return output

    
    # def post_process_single_image(self, image_path, mm_per_pixel, mask_eroded, mask_unet):
    #     orig_img = cv2.imread(image_path)
    #     if orig_img is None:
    #         raise ValueError(f"Cannot read image: {image_path}")
    #     mask_eroded  = 1 - mask_eroded
    #     mask_unet = cv2.resize(mask_unet, (orig_img.shape[1], orig_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    #     mask_check = mask_unet & mask_eroded
    #     mask_check = mask_check.astype(np.uint8)
    #     mask_check = cv2.erode(mask_check, np.ones((3, 3), np.uint8), iterations=2)
    #     mask_check = cv2.dilate(mask_check, np.ones((3, 3), np.uint8), iterations=2)
    #     mask_check = cv2.dilate(mask_check, np.ones((3, 3), np.uint8), iterations=1)
    #     mask_check = cv2.erode(mask_check, np.ones((3, 3), np.uint8), iterations=1)
    #     _, labeled_mask = cv2.connectedComponents(mask_check)
    #     labeled_mask = np.uint8(labeled_mask)

    #     contours, __ = cv2.findContours(labeled_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #     for contour in contours:
    #         min_rect = cv2.minAreaRect(contour)
    #         catelyzer_length = max(min_rect[1])
    #         catelyzer_length_mm = round(catelyzer_length * mm_per_pixel)
    #         rect_points = cv2.boxPoints(min_rect)
    #         rect_points = np.intp(rect_points)
    #         cv2.drawContours(orig_img, [rect_points], -1, (0, 0, 255), 2)
    #         cv2.putText(orig_img, str(catelyzer_length_mm), (int(min_rect[0][0]),  int(min_rect[0][1])), 0, 0.5, (0, 255, 0), 2, 16)
    #     return orig_img

    def post_process_single_image(self, image_path, mask_eroded, mask_unet):
        orig_img = cv2.imread(image_path)
        if orig_img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        mask_eroded  = 1 - mask_eroded
        mask_unet = cv2.resize(mask_unet, (orig_img.shape[1], orig_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_check = mask_unet & mask_eroded
        mask_check = mask_check.astype(np.uint8)
        mask_check = cv2.erode(mask_check, np.ones((3, 3), np.uint8), iterations=2)
        mask_check = cv2.dilate(mask_check, np.ones((3, 3), np.uint8), iterations=2)
        mask_check = cv2.dilate(mask_check, np.ones((3, 3), np.uint8), iterations=1)
        mask_check = cv2.erode(mask_check, np.ones((3, 3), np.uint8), iterations=1)
        _, labeled_mask = cv2.connectedComponents(mask_check)
        labeled_mask = np.uint8(labeled_mask)
        return labeled_mask


    def detect_size_outliers(self, contours):
        areas = [cv2.contourArea(contour) for contour in contours]
        # 使用Z-score方法检测异常值
        mean_area = np.mean(areas)
        std_area = np.std(areas)
        outlier_indices = []
        for i, area in enumerate(areas):
            z_score = abs(area - mean_area) / std_area if std_area > 0 else 0
            if z_score > self.config['size_outlier_factor']:
                outlier_indices.append(i)
        
        return outlier_indices


    def analyze_shape_features(self, contour: np.ndarray) -> Dict:
        # 基本几何特征
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # 外接矩形
        # x, y, w, h = cv2.boundingRect(contour)
        # aspect_ratio = float(w) / h if h > 0 else 0
        
        # 最小外接矩形
        rect = cv2.minAreaRect(contour)
        min_w, min_h = rect[1]
        min_aspect_ratio = max(min_w, min_h) / min(min_w, min_h) if min(min_w, min_h) > 0 else 0

        catelyzer_length = max(rect[1])
        catelyzer_length_mm = round(catelyzer_length * self.len_per_pixel)
        rect_points = cv2.boxPoints(rect)
        rect_points = np.intp(rect_points)
        
        # 圆形度 (4π*area/perimeter²)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # 凸包和凸度
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # 椭圆拟合
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            ellipse_aspect_ratio = max(ellipse[1]) / min(ellipse[1]) if min(ellipse[1]) > 0 else 0
        else:
            ellipse_aspect_ratio = 0
        
        return {
            'area': area,
            'perimeter': perimeter,
            # 'aspect_ratio': aspect_ratio,
            'min_aspect_ratio': min_aspect_ratio,
            'circularity': circularity,
            'solidity': solidity,
            'ellipse_aspect_ratio': ellipse_aspect_ratio,
            'bbox': rect_points
        }


    def detect_surface_defects(self, image: np.ndarray, contour: np.ndarray) -> Dict:
        # 创建掩模
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # 轮廓粗糙度分析
        hull = cv2.convexHull(contour)
        hull_perimeter = cv2.arcLength(hull, True)
        contour_perimeter = cv2.arcLength(contour, True)
        roughness = (contour_perimeter - hull_perimeter) / hull_perimeter if hull_perimeter > 0 else 0
        
        # 边缘方差分析
        # 计算轮廓上各点到中心的距离方差
        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            
            distances = []
            for point in contour.reshape(-1, 2):
                dist = np.sqrt((point[0] - cx)**2 + (point[1] - cy)**2)
                distances.append(dist)
            
            edge_variance = np.var(distances)
        else:
            edge_variance = 0
        
        # 检测凹陷缺陷
        defects = cv2.convexityDefects(contour, cv2.convexHull(contour, returnPoints=False))
        defect_count = len(defects) if defects is not None else 0
        
        return {
            'roughness': roughness,
            'edge_variance': edge_variance,
            'defect_count': defect_count
        }
        
    
    def classify_particle_type(self, is_size_outliers, shape_features, surface_features):
        # 判断是否为条状（正常催化剂）
        is_rod_like = (shape_features['min_aspect_ratio'] > self.config['aspect_ratio_threshold'] or
                      shape_features['ellipse_aspect_ratio'] > self.config['aspect_ratio_threshold'])
        
        # 判断是否为球形颗粒
        is_spherical = (shape_features['circularity'] > self.config['circularity_threshold'] and
                       shape_features['min_aspect_ratio'] < 1.5)
        
        # 判断是否有表面缺陷
        has_surface_defects = (surface_features['roughness'] > self.config['roughness_threshold'] or
                             surface_features['edge_variance'] > self.config['edge_variance_threshold'])
        
        # 分类逻辑
        if is_size_outliers:
            return 'size_outlier'
        else:
            if is_spherical:
                return 'spherical_particle'  # 球形异物颗粒
            elif is_rod_like and has_surface_defects:
                return 'defective_rod'       # 有缺陷的条状催化剂
            elif is_rod_like and not has_surface_defects:
                return 'normal_rod'          # 正常条状催化剂
            else:
                return 'irregular_particle'   # 不规则异物颗粒
    
    
    
    def detect_defects(self, image_path, net, device, len_per_pixel):
        _, mask_eroded = self.pre_process_single_image(image_path)
        mask_unet = self.inference_unet_batch(net, device, image_path)
        labeled_mask = self.post_process_single_image(image_path, mask_eroded, mask_unet)

        contours, __ = cv2.findContours(labeled_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {
                'total_particles': 0,
                'defective_particles': [],
                'particle_types': {},
                'size_outliers': []
            }
        # 分析每个颗粒
        results = {
            'total_particles': len(contours),
            'defective_particles': [],
            'particle_types': {
                'size_outlier': [],
                'normal_rod': [],
                'defective_rod': [],
                'spherical_particle': [],
                'irregular_particle': []
            },
            'size_outliers': []
        }
        
        # 检测大小异常
        size_outliers = self.detect_size_outliers(contours)
        results['size_outliers'] = size_outliers

        for i, contour in enumerate(contours):
            # 大小异常分析
            is_size_outliers = True if i in size_outliers else False
            # 形状特征分析
            shape_features = self.analyze_shape_features(contour)
            # 表面缺陷检测
            image = cv2.imread(image_path)
            surface_features = self.detect_surface_defects(image, contour)
            # 分类
            particle_type = self.classify_particle_type(is_size_outliers, shape_features, surface_features)

            # 存储结果
            particle_info = {
                'index': i,
                'contour': contour,
                'shape_features': shape_features,
                'surface_features': surface_features,
                # 'is_size_outlier': is_size_outliers
            }
            results['particle_types'][particle_type].append(particle_info)
            if particle_type != 'normal_rod':
                results['defective_particles'].append(particle_info)
        return results



        # vis_image = self.post_process_single_image(image_path, mm_per_pixel, mask_eroded, mask_unet)
        # filename = os.path.basename(image_path)
        # name, ext = os.path.splitext(filename)
        # output_path = os.path.join(output_dir, f"{name}_result{ext}")
        # cv2.imwrite(output_path, vis_image)
        
        # return True, output_path

    
    def visualize_results(self, imagepath, results, save_path: str = None):
        # 定义颜色
        result_img = cv2.imread(imagepath)
        if len(result_img.shape) == 2:
            result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)

        colors = {
            'size_outlier': (255, 255, 0),      # 青色 - 大小不正常
            'normal_rod': (0, 255, 0),          # 绿色 - 正常
            'defective_rod': (0, 165, 255),     # 橙色 - 有缺陷的条状
            'spherical_particle': (0, 0, 255),  # 红色 - 球形异物
            'irregular_particle': (255, 0, 255) # 紫色 - 不规则异物
        }
        for particle_type, particles in results['particle_types'].items():
            color = colors[particle_type]
            for particle in particles:
                contour = particle['contour']
                cv2.drawContours(result_img, [contour], -1, color, 2)
        
        legend_y = 30
        for particle_type, color in colors.items():
            count = len(results['particle_types'][particle_type])
            text = f"{particle_type}: {count}"
            cv2.putText(result_img, text, (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            legend_y += 25
        
        # 显示总统计
        total_defects = len(results['defective_particles'])
        defect_rate = (total_defects / results['total_particles'] * 100) if results['total_particles'] > 0 else 0
        
        cv2.putText(result_img, f"Total: {results['total_particles']}", (10, legend_y + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result_img, f"Defects: {total_defects} ({defect_rate:.1f}%)", (10, legend_y + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if save_path:
            filename = os.path.basename(imagepath)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(save_path, f"{name}_result{ext}")
            cv2.imwrite(output_path, result_img)
        
        return result_img



def main():
    args = parse_args()
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist!")
        return
    os.makedirs(args.output_dir, exist_ok=True)
    image_files = get_image_files(args.input_dir, args.image_exts)
    if not image_files:
        print(f"No image files found in '{args.input_dir}' with extensions: {args.image_exts}")
        return
    print(f"Found {len(image_files)} images to process")

    len_per_pixel = float(args.grid_len/ args.grid_pixel)
    detector = CatalystDefectDetector(len_per_pixel)

    print("Loading UNet model...")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    net = UNet(n_channels=3, n_classes=2, bilinear=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    
    try:
        state_dict = torch.load(args.model, map_location=device)
        _ = state_dict.pop('mask_values', [0, 1])
        net.load_state_dict(state_dict)
        print(f'Model loaded successfully! Using device: {device}')
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    successful = 0
    failed = 0
    failed_files = []

    print("Processing images...")
    for image_path in tqdm(image_files, desc="Processing", unit="image"):
        results = detector.detect_defects(image_path, net, device, len_per_pixel)
        result_img = detector.visualize_results(image_path, results, args.output_dir)
        
    #     if success:
    #         successful += 1
    #     else:
    #         failed += 1
    #         failed_files.append((image_path, result))
    #         print(f"\nFailed to process {image_path}: {result}")
    
    # print(f"\n{'='*50}")
    # print(f"Processing completed!")
    # print(f"Total images: {len(image_files)}")
    # print(f"Successful: {successful}")
    # print(f"Failed: {failed}")
    # print(f"Results saved to: {args.output_dir}")
    
    # if failed_files:
    #     print(f"\nFailed files:")
    #     for file_path, error in failed_files:
    #         print(f"  - {file_path}: {error}")

if __name__ == '__main__':
    main()