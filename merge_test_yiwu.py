#!/usr/bin/env python3
"""
催化剂异物异形检测系统

1. 图像预处理和UNet推理
2. 连通域分析和特征提取
3. 智能误报过滤
4. 异常分类（正常、异物、异形）
5. 可视化结果输出

"""

import os
import logging
import glob
import math
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import argparse

import cv2
import numpy as np
from tqdm import tqdm
import torch

from utils.data_loading import SelfDataset
from unet import UNet


# ============================================================================
# 配置类 - 集中管理所有常量和参数
# ============================================================================

@dataclass
class DetectionConfig:
    """检测算法配置参数类"""
    
    # 图像处理参数
    YUV_BRIGHTNESS_THRESHOLD: int = 15
    WHITE_PIXEL_VALUE: List[int] = None
    UNET_SCALE_FACTOR: float = 0.5
    
    # 连通域过滤参数
    MIN_CONTOUR_POINTS: int = 10
    CONTOUR_EPSILON_FACTOR: float = 0.005
    MIN_SIMPLIFIED_CONTOUR_POINTS: int = 5
    SKELETON_MIN_POINTS: int = 10
    SKELETON_MIN_ORDERED_POINTS: int = 5
    
    # 形态学参数
    EROSION_KERNEL_SIZE: Tuple[int, int] = (3, 3)
    DILATION_KERNEL_SIZE: Tuple[int, int] = (5, 5)
    OPEN_ITERATIONS: int = 1
    
    # 几何特征阈值
    MIN_CONTOUR_LENGTH: int = 3
    MIN_COMPONENT_FOR_ORIENTATION: int = 5
    ZERO_DIVISION_EPSILON: float = 1e-6
    CURVATURE_SCALE_FACTOR: float = 50
    CURVATURE_BASE_FACTOR: float = 20
    
    # 连通域合并参数
    MAX_AREA_RATIO_DIFF: float = 5.0
    ANGLE_CIRCULAR_THRESHOLD: float = 180.0
    
    # 误报检测参数
    COMPLEX_CONTOUR_THRESHOLD: int = 20
    
    # 统计分析参数
    MIN_SAMPLES_FOR_STATS: int = 3
    IQR_MULTIPLIER: float = 1.5
    CONSERVATIVE_THRESHOLD_MULTIPLIER_HIGH: float = 2.5
    CONSERVATIVE_THRESHOLD_MULTIPLIER_LOW: float = 0.4
    MEDIAN_MULTIPLIER_HIGH: float = 2.0
    MEDIAN_MULTIPLIER_LOW: float = 0.5
    OUTLIER_DETECTION_MULTIPLIER: float = 1.2
    OUTLIER_LOWER_BOUND_FACTOR: float = 0.3
    
    # 异常评分阈值
    AREA_TOLERANCE_FACTOR: float = 1.2
    ASPECT_RATIO_TOLERANCE: float = 0.8
    SOLIDITY_TOLERANCE: float = 0.8
    HIGH_CIRCULARITY_THRESHOLD: float = 0.8
    MEDIUM_CIRCULARITY_THRESHOLD: float = 0.7
    
    # 分类参数
    MIN_AREA_SMALL_FACTOR: float = 0.5
    MIN_AREA_MEDIUM_FACTOR: float = 1.5
    
    # 可视化参数
    FONT_SCALE: float = 0.6
    FONT_THICKNESS: int = 2
    LINE_SPACING: int = 8
    BACKGROUND_PADDING: int = 5
    DETAIL_FONT_SCALE: float = 0.5
    DETAIL_FONT_THICKNESS: int = 1
    FALSE_POSITIVE_FONT_SCALE: float = 0.8
    FALSE_POSITIVE_FONT_THICKNESS: int = 2
    NORMAL_FONT_SCALE: float = 0.5
    NORMAL_FONT_THICKNESS: int = 1
    VISUALIZATION_ALPHA: float = 0.8
    VISUALIZATION_BETA: float = 0.2
    
    # 颜色定义 (BGR格式)
    COLOR_FOREIGN_OBJECTS: Tuple[int, int, int] = (0, 0, 255)      # 红色
    COLOR_DEFORMED_CATALYSTS: Tuple[int, int, int] = (0, 128, 255) # 橙色
    COLOR_NORMAL: Tuple[int, int, int] = (0, 255, 0)              # 绿色
    COLOR_FALSE_POSITIVE: Tuple[int, int, int] = (128, 0, 128)    # 紫色
    COLOR_WHITE: Tuple[int, int, int] = (255, 255, 255)           # 白色
    COLOR_BLACK: Tuple[int, int, int] = (0, 0, 0)                 # 黑色
    
    def __post_init__(self):
        """初始化后处理"""
        if self.WHITE_PIXEL_VALUE is None:
            self.WHITE_PIXEL_VALUE = [255, 255, 255]


# ============================================================================
# 异常处理类
# ============================================================================

class DetectionError(Exception):
    """检测算法基础异常类"""
    pass


class ImageProcessingError(DetectionError):
    """图像处理异常"""
    pass


class ModelInferenceError(DetectionError):
    """模型推理异常"""
    pass


class ComponentAnalysisError(DetectionError):
    """连通域分析异常"""
    pass


# ============================================================================
# 日志配置
# ============================================================================

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """配置日志系统"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# ============================================================================
# 工具函数模块
# ============================================================================

class MathUtils:
    """数学计算工具类"""
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float, 
                   default: float = 0.0) -> float:
        """安全除法，避免除零错误"""
        return numerator / denominator if abs(denominator) > DetectionConfig.ZERO_DIVISION_EPSILON else default
    
    @staticmethod
    def calculate_distance(point1: Tuple[float, float], 
                          point2: Tuple[float, float]) -> float:
        """计算两点间欧氏距离"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    @staticmethod
    def normalize_angle(angle: float) -> float:
        """标准化角度到-45到45度之间"""
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90
        return angle


class GeometryUtils:
    """几何计算工具类"""
    
    @staticmethod
    def calculate_aspect_ratio(width: float, height: float) -> float:
        """计算长宽比"""
        if width == 0 or height == 0:
            return 1.0
        return max(width, height) / min(width, height)
    
    @staticmethod
    def calculate_circularity(area: float, perimeter: float) -> float:
        """计算圆形度"""
        if perimeter <= 0:
            return 0.0
        return 4 * np.pi * area / (perimeter * perimeter)
    
    @staticmethod
    def calculate_solidity(contour_area: float, hull_area: float) -> float:
        """计算实心度"""
        return MathUtils.safe_divide(contour_area, hull_area, 0.0)


class StatisticsUtils:
    """统计分析工具类"""
    
    @staticmethod
    def calculate_outlier_thresholds(data: np.ndarray) -> Tuple[float, float, Dict[str, float]]:
        """
        计算离群值检测阈值
        
        Args:
            data: 输入数据数组
            
        Returns:
            Tuple[lower_threshold, upper_threshold, stats_dict]
        """
        config = DetectionConfig()
        
        if len(data) < config.MIN_SAMPLES_FOR_STATS:
            mean_val = np.mean(data) if len(data) > 0 else 10
            upper_threshold = mean_val * config.CONSERVATIVE_THRESHOLD_MULTIPLIER_HIGH
            lower_threshold = mean_val * config.CONSERVATIVE_THRESHOLD_MULTIPLIER_LOW
            stats = {
                'median': mean_val,
                'q25': mean_val,
                'q75': mean_val,
                'iqr': 0,
                'sample_count': len(data)
            }
        else:
            median_val = np.median(data)
            q75 = np.percentile(data, 75)
            q25 = np.percentile(data, 25)
            iqr = q75 - q25
            
            if iqr > 0:
                upper_threshold = q75 + config.IQR_MULTIPLIER * iqr
                lower_threshold = max(
                    float(q25 - config.IQR_MULTIPLIER * iqr),
                    float(median_val * config.OUTLIER_LOWER_BOUND_FACTOR)
                )
            else:
                upper_threshold = median_val * config.MEDIAN_MULTIPLIER_HIGH
                lower_threshold = median_val * config.MEDIAN_MULTIPLIER_LOW
                
            stats = {
                'median': median_val,
                'q25': q25,
                'q75': q75,
                'iqr': iqr,
                'sample_count': len(data)
            }
        
        return lower_threshold, upper_threshold, stats


# ============================================================================
# 图像处理模块
# ============================================================================

class ImageProcessor:
    """图像处理器类"""
    
    def __init__(self, config: DetectionConfig = None):
        self.config = config or DetectionConfig()
        
    def preprocess_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        图像预处理：处理暗区域
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            Tuple[processed_image, mask_eroded]
            
        Raises:
            ImageProcessingError: 图像处理失败时抛出
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ImageProcessingError(f"无法读取图像: {image_path}")
            
            # YUV色彩空间转换，过滤低亮度区域
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            mask_filter = yuv[:, :, 0] < self.config.YUV_BRIGHTNESS_THRESHOLD
            mask_filter = mask_filter.astype(np.uint8)
            
            # 形态学操作清理掩码
            kernel = np.ones((5, 5), np.uint8)
            mask_eroded = cv2.dilate(mask_filter, kernel, iterations=2)
            mask_eroded = cv2.erode(mask_eroded, kernel, iterations=2)
            
            # 将暗区域设置为白色
            image[mask_eroded == 1] = self.config.WHITE_PIXEL_VALUE
            
            logger.debug(f"成功预处理图像: {image_path}")
            return image, mask_eroded
            
        except Exception as e:
            raise ImageProcessingError(f"图像预处理失败: {str(e)}")
    
    def apply_morphological_operations(self, mask: np.ndarray) -> np.ndarray:
        """
        应用形态学操作清理掩码
        
        Args:
            mask: 输入掩码
            
        Returns:
            清理后的掩码
        """
        try:
            # 腐蚀和膨胀操作
            erosion_kernel = np.ones(self.config.EROSION_KERNEL_SIZE, np.uint8)
            mask_clean = cv2.erode(mask, erosion_kernel, iterations=1)
            mask_clean = cv2.dilate(mask_clean, erosion_kernel, iterations=1)
            
            # 开运算去除小噪声点
            open_kernel = np.ones(self.config.DILATION_KERNEL_SIZE, np.uint8)
            mask_clean = cv2.morphologyEx(
                mask_clean, cv2.MORPH_OPEN, open_kernel, 
                iterations=self.config.OPEN_ITERATIONS
            )
            
            return mask_clean
            
        except Exception as e:
            logger.error(f"形态学操作失败: {str(e)}")
            return mask


# ============================================================================
# 模型推理模块
# ============================================================================

class UNetInference:
    """UNet模型推理器"""
    
    def __init__(self, config: DetectionConfig = None):
        self.config = config or DetectionConfig()
        
    def inference(self, net: torch.nn.Module, device: torch.device, 
                 image_path: str) -> np.ndarray:
        """
        UNet模型推理
        
        Args:
            net: UNet网络模型
            device: 计算设备
            image_path: 图像路径
            
        Returns:
            推理结果掩码
            
        Raises:
            ModelInferenceError: 模型推理失败时抛出
        """
        try:
            img_ori = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img_ori is None:
                raise ModelInferenceError(f"无法读取图像: {image_path}")
            
            net.eval()
            img = torch.from_numpy(
                SelfDataset.preprocess(None, img_ori, self.config.UNET_SCALE_FACTOR, is_mask=False)
            )
            img = img.unsqueeze(0)
            img = img.to(device=device, dtype=torch.float32)
            
            with torch.no_grad():
                output = net(img).cpu()
                mask = output.argmax(dim=1)
                output = (mask[0] * 255).squeeze().numpy().astype(np.uint8)
            
            logger.debug(f"成功完成UNet推理: {image_path}")
            return output
            
        except Exception as e:
            raise ModelInferenceError(f"UNet推理失败: {str(e)}")


# ============================================================================
# 连通域分析模块
# ============================================================================

class ComponentAnalyzer:
    """连通域分析器"""
    
    def __init__(self, config: DetectionConfig = None):
        self.config = config or DetectionConfig()
        
    def filter_small_components(self, mask: np.ndarray, min_area: int) -> np.ndarray:
        """
        过滤掉面积过小的连通域
        
        Args:
            mask: 输入掩码
            min_area: 最小面积阈值
            
        Returns:
            过滤后的掩码
        """
        try:
            num_labels, labeled_mask = cv2.connectedComponents(mask)
            filtered_mask = np.zeros_like(mask)
            
            for label in range(1, num_labels):
                component_mask = (labeled_mask == label).astype(np.uint8)
                area = cv2.countNonZero(component_mask)
                
                if area >= min_area:
                    filtered_mask[component_mask > 0] = 255
            
            return filtered_mask
            
        except Exception as e:
            logger.error(f"连通域过滤失败: {str(e)}")
            return mask
    
    def analyze_connected_components(self, mask: np.ndarray) -> List[Dict[str, Any]]:
        """
        连通域分析和特征提取
        
        Args:
            mask: 输入掩码
            
        Returns:
            连通域信息列表
        """
        try:
            num_labels, labeled_mask = cv2.connectedComponents(mask)
            components_info = []
            
            for label in range(1, num_labels):
                component_info = self._extract_component_features(labeled_mask, label)
                if component_info:
                    components_info.append(component_info)
            
            logger.debug(f"分析得到 {len(components_info)} 个连通域")
            return components_info
            
        except Exception as e:
            raise ComponentAnalysisError(f"连通域分析失败: {str(e)}")
    
    def _extract_component_features(self, labeled_mask: np.ndarray, 
                                  label: int) -> Optional[Dict[str, Any]]:
        """
        提取单个连通域的特征
        
        Args:
            labeled_mask: 标记掩码
            label: 连通域标签
            
        Returns:
            连通域特征字典
        """
        try:
            component_mask = (labeled_mask == label).astype(np.uint8)
            contours, _ = cv2.findContours(
                component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            if len(contours) == 0:
                return None
                
            contour = contours[0]
            area = cv2.contourArea(contour)
            
            # 最小外接矩形
            min_rect = cv2.minAreaRect(contour)
            width, height = min_rect[1]
            if width == 0 or height == 0:
                return None
            
            # 几何特征计算
            aspect_ratio = GeometryUtils.calculate_aspect_ratio(width, height)
            x, y, w, h = cv2.boundingRect(contour)
            
            # 实心度计算
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = GeometryUtils.calculate_solidity(area, hull_area)
            
            # 圆形度计算
            perimeter = cv2.arcLength(contour, True)
            circularity = GeometryUtils.calculate_circularity(area, perimeter)
            
            # 密度特征
            min_rect_area = width * height
            bbox_density = MathUtils.safe_divide(area, min_rect_area)
            
            # 复杂度评分
            complexity_score = self._calculate_complexity_score(contour, hull)
            
            # 中心点计算
            center_x, center_y = self._calculate_centroid(contour, x, y, w, h)
            
            return {
                'label': label,
                'area': area,
                'aspect_ratio': aspect_ratio,
                'solidity': solidity,
                'circularity': circularity,
                'center': (center_x, center_y),
                'bbox': (x, y, w, h),
                'min_rect': min_rect,
                'contour': contour,
                'mask': component_mask,
                'anomaly_score': 0,
                'anomaly_reasons': [],
                'bbox_density': bbox_density,
                'complexity_score': complexity_score
            }
            
        except Exception as e:
            logger.error(f"特征提取失败: {str(e)}")
            return None
    
    def _calculate_complexity_score(self, contour: np.ndarray, 
                                  hull: np.ndarray) -> int:
        """计算轮廓复杂度评分"""
        try:
            hull_indices = cv2.convexHull(contour, returnPoints=False)
            if len(hull_indices) > 3 and len(contour) > 3:
                defects = cv2.convexityDefects(contour, hull_indices)
                return len(defects) if defects is not None else 0
            return 0
        except Exception:
            return 0
    
    def _calculate_centroid(self, contour: np.ndarray, x: int, y: int, 
                          w: int, h: int) -> Tuple[int, int]:
        """计算连通域中心点"""
        try:
            moments = cv2.moments(contour)
            if moments['m00'] != 0:
                center_x = int(moments['m10'] / moments['m00'])
                center_y = int(moments['m01'] / moments['m00'])
                return center_x, center_y
            else:
                return x + w // 2, y + h // 2
        except Exception:
            return x + w // 2, y + h // 2


# ============================================================================
# 曲率分析模块
# ============================================================================

class CurvatureAnalyzer:
    """弯曲度分析器"""
    
    def __init__(self, config: DetectionConfig = None):
        self.config = config or DetectionConfig()
    
    def calculate_curvature_features(self, contour: np.ndarray, 
                                   component_mask: np.ndarray) -> Dict[str, float]:
        """
        计算弯曲度特征
        
        Args:
            contour: 轮廓
            component_mask: 连通域掩码
            
        Returns:
            弯曲度特征字典
        """
        if len(contour) < self.config.MIN_CONTOUR_POINTS:
            return {
                'skeleton_curvature': 0,
                'straightness_ratio': 1.0
            }
        
        # 简化轮廓
        simplified_contour = self._simplify_contour(contour)
        
        # 计算骨架弯曲度
        skeleton_curvature = self._calculate_skeleton_curvature(component_mask)
        
        # 计算直线度比例
        straightness_ratio = self._calculate_straightness_ratio(simplified_contour)
        
        return {
            'skeleton_curvature': skeleton_curvature,
            'straightness_ratio': straightness_ratio
        }
    
    def _simplify_contour(self, contour: np.ndarray) -> np.ndarray:
        """简化轮廓"""
        epsilon = self.config.CONTOUR_EPSILON_FACTOR * cv2.arcLength(contour, True)
        simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(simplified_contour) < self.config.MIN_SIMPLIFIED_CONTOUR_POINTS:
            return contour
        return simplified_contour
    
    def _calculate_skeleton_curvature(self, component_mask: np.ndarray) -> float:
        """计算骨架线弯曲度"""
        try:
            if hasattr(cv2, 'ximgproc'):
                skeleton = cv2.ximgproc.thinning(component_mask)
                return self._analyze_skeleton(skeleton)
            else:
                return self._calculate_simplified_skeleton_curvature(component_mask)
        except Exception:
            return self._calculate_simplified_skeleton_curvature(component_mask)
    
    def _analyze_skeleton(self, skeleton: np.ndarray) -> float:
        """分析骨架线"""
        skeleton_points = np.column_stack(np.where(skeleton > 0))
        
        if len(skeleton_points) < self.config.SKELETON_MIN_POINTS:
            return 0
        
        ordered_points = self._order_skeleton_points(skeleton_points)
        
        if len(ordered_points) < self.config.SKELETON_MIN_ORDERED_POINTS:
            return 0
        
        # 计算总曲率
        total_curvature = 0
        for i in range(1, len(ordered_points) - 1):
            p1, p2, p3 = ordered_points[i-1], ordered_points[i], ordered_points[i+1]
            curvature = self._calculate_point_curvature(p1, p2, p3)
            total_curvature += curvature
        
        return MathUtils.safe_divide(total_curvature, len(ordered_points))
    
    def _calculate_simplified_skeleton_curvature(self, component_mask: np.ndarray) -> float:
        """简化版骨架弯曲度计算"""
        try:
            # 距离变换
            dist_transform = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)
            
            # 找到所有非零像素点
            nonzero_points = np.column_stack(np.where(component_mask > 0))
            
            if len(nonzero_points) < self.config.SKELETON_MIN_POINTS:
                return 0
            
            # 主成分分析
            centered_points = nonzero_points - np.mean(nonzero_points, axis=0)
            cov_matrix = np.cov(centered_points.T)
            eigenvalues = np.linalg.eigvals(cov_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]
            
            if eigenvalues[0] > self.config.ZERO_DIVISION_EPSILON:
                axis_ratio = eigenvalues[1] / eigenvalues[0]
                
                # 使用凸包面积比例
                contours, _ = cv2.findContours(
                    component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if len(contours) > 0:
                    contour = contours[0]
                    contour_area = cv2.contourArea(contour)
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    
                    if hull_area > 0:
                        solidity = contour_area / hull_area
                        return axis_ratio * (1 - solidity) * self.config.CURVATURE_SCALE_FACTOR
                
                return axis_ratio * self.config.CURVATURE_BASE_FACTOR
            
            return 0
        except Exception:
            return 0
    
    def _order_skeleton_points(self, skeleton_points: np.ndarray) -> np.ndarray:
        """对骨架点进行空间排序"""
        if len(skeleton_points) < 3:
            return skeleton_points
        
        # 找到最远的两个点作为端点
        max_dist = 0
        start_idx, end_idx = 0, 0
        
        for i in range(len(skeleton_points)):
            for j in range(i + 1, len(skeleton_points)):
                dist = np.linalg.norm(skeleton_points[i] - skeleton_points[j])
                if dist > max_dist:
                    max_dist = dist
                    start_idx, end_idx = i, j
        
        start_point = skeleton_points[start_idx]
        end_point = skeleton_points[end_idx]
        
        # 计算主方向
        direction = end_point - start_point
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm == 0:
            return skeleton_points
        
        direction = direction / direction_norm
        
        # 按投影值排序
        projections = []
        for point in skeleton_points:
            proj = np.dot(point - start_point, direction)
            projections.append(proj)
        
        sorted_indices = np.argsort(projections)
        return skeleton_points[sorted_indices]
    
    def _calculate_point_curvature(self, p1: np.ndarray, p2: np.ndarray, 
                                 p3: np.ndarray) -> float:
        """计算三点构成的曲率"""
        v1 = p2 - p1
        v2 = p3 - p2
        
        d1 = np.linalg.norm(v1)
        d2 = np.linalg.norm(v2)
        
        if d1 == 0 or d2 == 0:
            return 0
        
        cross_product = np.cross(v1, v2)
        area = abs(cross_product) / 2.0
        
        # 曲率计算
        denominator = d1 * d2 * np.linalg.norm(p3 - p1)
        return MathUtils.safe_divide(4 * area, denominator)
    
    def _calculate_straightness_ratio(self, contour: np.ndarray) -> float:
        """计算直线度比例"""
        if len(contour) < self.config.MIN_CONTOUR_LENGTH:
            return 1.0
        
        points = contour.reshape(-1, 2)
        start_point = points[0]
        end_point = points[-1]
        straight_distance = np.linalg.norm(end_point - start_point)
        
        contour_length = cv2.arcLength(contour, False)
        return MathUtils.safe_divide(straight_distance, contour_length, 1.0)


# ============================================================================
# 连通域合并模块
# ============================================================================

class ComponentMerger:
    """连通域合并器"""
    
    def __init__(self, config: DetectionConfig = None):
        self.config = config or DetectionConfig()
    
    def merge_connected_components(self, components_info: List[Dict[str, Any]], 
                                 merge_distance: float, 
                                 angle_threshold: float) -> List[Dict[str, Any]]:
        """
        智能合并连通域
        
        Args:
            components_info: 连通域信息列表
            merge_distance: 合并距离阈值
            angle_threshold: 角度阈值
            
        Returns:
            合并后的连通域列表
        """
        if not components_info:
            return components_info
        
        # 使用并查集进行合并
        n = len(components_info)
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # 检查每对连通域是否需要合并
        for i in range(n):
            for j in range(i + 1, n):
                if self._should_merge_components(
                    components_info[i], components_info[j], 
                    merge_distance, angle_threshold
                ):
                    union(i, j)
        
        # 按合并组重新组织
        groups = {}
        for i in range(n):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        
        # 合并每个组
        merged_components = []
        for group_indices in groups.values():
            if len(group_indices) == 1:
                merged_components.append(components_info[group_indices[0]])
            else:
                merged_component = self._merge_component_group(
                    [components_info[i] for i in group_indices]
                )
                merged_components.append(merged_component)
        
        logger.debug(f"连通域合并: {len(components_info)} -> {len(merged_components)}")
        return merged_components
    
    def _should_merge_components(self, comp1: Dict[str, Any], comp2: Dict[str, Any], 
                               merge_distance: float, angle_threshold: float) -> bool:
        """判断两个连通域是否应该合并"""
        # 距离检查
        center1, center2 = comp1['center'], comp2['center']
        distance = MathUtils.calculate_distance(center1, center2)
        
        if distance > merge_distance:
            return False
        
        # 角度检查
        angle1 = self._calculate_component_orientation(comp1['contour'])
        angle2 = self._calculate_component_orientation(comp2['contour'])
        angle_diff = abs(angle1 - angle2)
        angle_diff = min(angle_diff, self.config.ANGLE_CIRCULAR_THRESHOLD - angle_diff)
        
        if angle_diff > angle_threshold:
            return False
        
        # 面积比例检查
        area1, area2 = comp1['area'], comp2['area']
        area_ratio = max(area1, area2) / min(area1, area2)
        
        return area_ratio <= self.config.MAX_AREA_RATIO_DIFF
    
    def _calculate_component_orientation(self, contour: np.ndarray) -> float:
        """计算连通域主方向角度"""
        if len(contour) < self.config.MIN_COMPONENT_FOR_ORIENTATION:
            return 0
        
        min_rect = cv2.minAreaRect(contour)
        angle = min_rect[2]
        return MathUtils.normalize_angle(angle)
    
    def _merge_component_group(self, component_group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """合并一组连通域"""
        if len(component_group) == 1:
            return component_group[0]
        
        # 合并轮廓点
        all_points = []
        total_area = 0
        
        for comp in component_group:
            all_points.extend(comp['contour'].reshape(-1, 2))
            total_area += comp['area']
        
        # 重新计算特征
        all_points = np.array(all_points)
        hull = cv2.convexHull(all_points.reshape(-1, 1, 2))
        
        area = total_area
        min_rect = cv2.minAreaRect(hull)
        width, height = min_rect[1]
        
        aspect_ratio = GeometryUtils.calculate_aspect_ratio(width, height)
        x, y, w, h = cv2.boundingRect(hull)
        
        hull_area = cv2.contourArea(hull)
        solidity = GeometryUtils.calculate_solidity(area, hull_area)
        
        perimeter = cv2.arcLength(hull, True)
        circularity = GeometryUtils.calculate_circularity(area, perimeter)
        
        # 计算中心点
        moments = cv2.moments(hull)
        if moments['m00'] != 0:
            center_x = int(moments['m10'] / moments['m00'])
            center_y = int(moments['m01'] / moments['m00'])
        else:
            center_x, center_y = x + w // 2, y + h // 2
        
        # 合并mask
        merged_mask = np.zeros_like(component_group[0]['mask'])
        for comp in component_group:
            merged_mask = cv2.bitwise_or(merged_mask, comp['mask'])
        
        # 合并异常信息
        max_anomaly_score = max(
            comp.get('anomaly_score', 0) for comp in component_group
        )
        combined_reasons = []
        for comp in component_group:
            combined_reasons.extend(comp.get('anomaly_reasons', []))
        unique_reasons = list(set(combined_reasons))
        
        # 密度特征
        min_rect_area = width * height
        bbox_density = MathUtils.safe_divide(area, min_rect_area)
        
        # 复杂度评分
        complexity_score = 0
        try:
            hull_indices = cv2.convexHull(hull, returnPoints=False)
            if len(hull_indices) > 3 and len(hull) > 3:
                defects = cv2.convexityDefects(hull, hull_indices)
                complexity_score = len(defects) if defects is not None else 0
        except Exception:
            complexity_score = 0
        
        return {
            'label': component_group[0]['label'],
            'area': area,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity,
            'circularity': circularity,
            'center': (center_x, center_y),
            'bbox': (x, y, w, h),
            'min_rect': min_rect,
            'contour': hull,
            'mask': merged_mask,
            'anomaly_score': max_anomaly_score,
            'anomaly_reasons': unique_reasons,
            'bbox_density': bbox_density,
            'complexity_score': complexity_score
        }


# ============================================================================
# 误报过滤模块
# ============================================================================

class FalsePositiveFilter:
    """误报过滤器"""
    
    def __init__(self, config: DetectionConfig = None):
        self.config = config or DetectionConfig()
    
    def filter_false_positives(self, components_info: List[Dict[str, Any]], 
                             args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        智能误报过滤
        
        Args:
            components_info: 连通域信息列表
            args: 命令行参数
            
        Returns:
            Tuple[filtered_components, false_positive_regions]
        """
        filtered_components = []
        false_positive_regions = []
        removed_count = 0
        
        logger.info(f"开始智能连通域过滤，初始连通域数量: {len(components_info)}")
        logger.info(f"误报区域可视化: {'启用' if args.show_false_positive else '禁用'}")
        
        for comp in components_info:
            if self._is_false_positive_region(comp, args):
                logger.debug(
                    f"检测到误报大区域: 面积={comp['area']}, "
                    f"密度={comp['bbox_density']:.3f}, "
                    f"复杂度={comp['complexity_score']}"
                )
                
                false_positive_regions.append({
                    'mask': comp['mask'],
                    'contour': comp['contour'],
                    'area': comp['area'],
                    'density': comp['bbox_density'],
                    'complexity': comp['complexity_score']
                })
                
                removed_count += 1
            else:
                filtered_components.append(comp)
        
        logger.info(
            f"智能过滤完成: 保留正常组件 {len(filtered_components)} 个，"
            f"去除误报区域 {removed_count} 个"
        )
        
        return filtered_components, false_positive_regions
    
    def _is_false_positive_region(self, component_info: Dict[str, Any], 
                                args: argparse.Namespace) -> bool:
        """判断连通域是否为误报区域"""
        area = component_info['area']
        bbox_density = component_info['bbox_density']
        complexity_score = component_info['complexity_score']
        
        # 判断条件
        is_oversized = area > args.fp_area_threshold
        is_low_density = bbox_density < args.fp_density_threshold
        is_complex = complexity_score > self.config.COMPLEX_CONTOUR_THRESHOLD
        
        # 综合评分
        false_positive_score = 0
        if is_oversized:
            false_positive_score += 1
        if is_low_density:
            false_positive_score += 3
        if is_complex:
            false_positive_score += 2
        
        return false_positive_score >= args.fp_score_threshold


# ============================================================================
# 异常分类模块
# ============================================================================

class AnomalyClassifier:
    """异常分类器"""
    
    def __init__(self, config: DetectionConfig = None):
        self.config = config or DetectionConfig()
    
    def classify_anomalies(self, components_info: List[Dict[str, Any]], 
                         image_shape: Tuple[int, int], 
                         args: argparse.Namespace) -> Dict[str, List[Dict[str, Any]]]:
        """
        异常区域分类
        
        Args:
            components_info: 连通域信息列表
            image_shape: 图像形状
            args: 命令行参数
            
        Returns:
            分类结果字典
        """
        normal_components = []
        foreign_objects = []
        deformed_catalysts = []
        
        # 计算短边分布统计
        short_sides = self._extract_short_sides(components_info)
        outlier_low, outlier_high, stats = StatisticsUtils.calculate_outlier_thresholds(
            np.array(short_sides)
        )
        
        self._log_statistics(stats, outlier_low, outlier_high)
        
        # 逐个分析连通域
        for comp in components_info:
            anomaly_score, anomaly_reasons = self._analyze_component_anomalies(
                comp, args, outlier_high
            )
            
            comp['anomaly_score'] = anomaly_score
            comp['anomaly_reasons'] = anomaly_reasons
            
            # 分类决策
            category = self._classify_component(comp, anomaly_score, anomaly_reasons, args)
            
            if category == 'normal':
                normal_components.append(comp)
            elif category == 'foreign':
                foreign_objects.append(comp)
            else:  # deformed
                deformed_catalysts.append(comp)
        
        return {
            'normal': normal_components,
            'foreign_objects': foreign_objects,
            'deformed_catalysts': deformed_catalysts
        }
    
    def _extract_short_sides(self, components_info: List[Dict[str, Any]]) -> List[float]:
        """提取短边长度列表"""
        short_sides = []
        for comp in components_info:
            min_rect = comp['min_rect']
            width_rect, height_rect = min_rect[1]
            short_side = min(width_rect, height_rect)
            short_sides.append(short_side)
        return short_sides
    
    def _log_statistics(self, stats: Dict[str, float], outlier_low: float, 
                       outlier_high: float) -> None:
        """记录统计信息"""
        if stats['sample_count'] >= self.config.MIN_SAMPLES_FOR_STATS:
            logger.info(
                f"当前图片短边分布统计: 连通域数={stats['sample_count']}, "
                f"中位数={stats['median']:.1f}, Q25={stats['q25']:.1f}, "
                f"Q75={stats['q75']:.1f}, IQR={stats['iqr']:.1f}"
            )
        else:
            logger.info(
                f"当前图片连通域数量较少({stats['sample_count']})，使用保守阈值"
            )
        
        logger.info(f"当前图片离群值阈值: 过细<{outlier_low:.1f}, 过粗>{outlier_high:.1f}")
    
    def _analyze_component_anomalies(self, comp: Dict[str, Any], 
                                   args: argparse.Namespace, 
                                   outlier_high: float) -> Tuple[int, List[str]]:
        """分析连通域异常"""
        anomaly_score = 0
        anomaly_reasons = []
        
        # 1. 面积异常检测
        if comp['area'] > args.max_area * self.config.AREA_TOLERANCE_FACTOR:
            anomaly_score += 2
            anomaly_reasons.append('area is too large')
        
        # 2. 长宽比异常检测
        if comp['aspect_ratio'] < args.min_aspect_ratio * self.config.ASPECT_RATIO_TOLERANCE:
            anomaly_score += 2
            anomaly_reasons.append('aspect ratio is too small')
        elif comp['aspect_ratio'] > args.max_aspect_ratio * self.config.AREA_TOLERANCE_FACTOR:
            anomaly_score += 2
            anomaly_reasons.append('aspect ratio is too large')
        elif comp['aspect_ratio'] < args.min_aspect_ratio:
            anomaly_score += 1
            anomaly_reasons.append('aspect ratio is slightly small')
        
        # 3. 实心度异常检测
        if comp['solidity'] < args.min_solidity * self.config.SOLIDITY_TOLERANCE:
            anomaly_score += 2
            anomaly_reasons.append('shape is irregular')
        elif comp['solidity'] < args.min_solidity:
            anomaly_score += 1
            anomaly_reasons.append('shape is slightly irregular')
        
        # 4. 圆形度异常检测
        if comp['circularity'] > self.config.HIGH_CIRCULARITY_THRESHOLD:
            anomaly_score += 2
            anomaly_reasons.append('shape is too circular')
        elif comp['circularity'] > self.config.MEDIUM_CIRCULARITY_THRESHOLD:
            anomaly_score += 1
            anomaly_reasons.append('shape is slightly circular')
        
        # 5. 短边离群检测
        min_rect = comp['min_rect']
        width_rect, height_rect = min_rect[1]
        component_short_side = min(width_rect, height_rect)
        
        if (component_short_side > self.config.OUTLIER_DETECTION_MULTIPLIER * outlier_high and 
            comp['bbox_density'] > args.fp_density_threshold):
            anomaly_score += 3
            anomaly_reasons.append('short side is too thick (outlier)')
            logger.debug(
                f"检测到过粗组件: 短边={component_short_side:.1f} > "
                f"当前图片阈值{outlier_high:.1f}"
            )
        
        return anomaly_score, anomaly_reasons
    
    def _classify_component(self, comp: Dict[str, Any], anomaly_score: int, 
                          anomaly_reasons: List[str], 
                          args: argparse.Namespace) -> str:
        """分类连通域"""
        if anomaly_score <= 1:
            return 'normal'
        elif anomaly_score == 2:
            # 保守分类
            if (comp['circularity'] > self.config.HIGH_CIRCULARITY_THRESHOLD or
                comp['area'] < args.min_area * self.config.MIN_AREA_SMALL_FACTOR or
                'short side is too thick (outlier)' in anomaly_reasons):
                return 'foreign'
            else:
                return 'normal'
        elif 3 <= anomaly_score <= 6:
            # 明显异常
            if comp['area'] < args.min_area * self.config.MIN_AREA_MEDIUM_FACTOR:
                return 'normal'
            elif (comp['circularity'] > self.config.MEDIUM_CIRCULARITY_THRESHOLD or
                  'shape is too circular' in anomaly_reasons or
                  'short side is too thick (outlier)' in anomaly_reasons):
                return 'foreign'
            else:
                return 'deformed'
        else:  # 高异常分数
            if (comp['area'] < args.min_area * self.config.MIN_AREA_MEDIUM_FACTOR or
                comp['circularity'] > self.config.MEDIUM_CIRCULARITY_THRESHOLD or
                'shape is too circular' in anomaly_reasons or
                'short side is too thick (outlier)' in anomaly_reasons):
                return 'foreign'
            else:
                return 'deformed'


# ============================================================================
# 可视化模块
# ============================================================================

class ResultVisualizer:
    """结果可视化器"""
    
    def __init__(self, config: DetectionConfig = None):
        self.config = config or DetectionConfig()
        
        self.colors = {
            'foreign_objects': self.config.COLOR_FOREIGN_OBJECTS,
            'deformed_catalysts': self.config.COLOR_DEFORMED_CATALYSTS,
            'normal': self.config.COLOR_NORMAL,
        }
        
        self.labels = {
            'foreign_objects': 'foreign_objects',
            'deformed_catalysts': 'deformed_catalysts',
            'normal': 'normal',
        }
    
    def visualize_results(self, original_image: np.ndarray, 
                         classification_result: Dict[str, List[Dict[str, Any]]], 
                         anomaly_mask: np.ndarray, 
                         false_positive_regions: Optional[List[Dict[str, Any]]] = None, 
                         show_false_positive: bool = False,
                         show_normal_density: bool = False) -> np.ndarray:
        """
        生成可视化结果
        
        Args:
            original_image: 原始图像
            classification_result: 分类结果
            anomaly_mask: 异常掩码
            false_positive_regions: 误报区域列表
            show_false_positive: 是否显示误报区域
            show_normal_density: 是否显示正常催化剂的density信息
            
        Returns:
            可视化图像
        """
        vis_image = original_image.copy()
        colored_mask = np.zeros_like(original_image)
        
        # 绘制分类结果
        for category, components in classification_result.items():
            if category not in self.colors:
                continue
                
            color = self.colors[category]
            
            for comp in components:
                # 填充连通域区域
                colored_mask[comp['mask'] > 0] = color
                
                # 绘制最小外接矩形
                self._draw_bounding_rect(vis_image, comp['min_rect'], color)
                
                # 添加标签信息
                if category in ['foreign_objects', 'deformed_catalysts']:
                    self._add_component_labels(vis_image, comp, color)
                elif category == 'normal' and show_normal_density:
                    self._add_normal_component_labels(vis_image, comp, color)
        
        # 绘制误报区域
        if show_false_positive and false_positive_regions:
            vis_image = self._draw_false_positive_regions(vis_image, false_positive_regions)
        
        return vis_image
    
    def _draw_bounding_rect(self, image: np.ndarray, min_rect: Tuple, color: Tuple[int, int, int]) -> None:
        """绘制最小外接矩形"""
        rect_points = cv2.boxPoints(min_rect)
        rect_points = np.int32(rect_points).reshape((-1, 1, 2))
        cv2.drawContours(image, [rect_points], -1, color, 2)
    
    def _add_component_labels(self, image: np.ndarray, comp: Dict[str, Any], 
                             color: Tuple[int, int, int]) -> None:
        """添加组件标签"""
        center_x, center_y = comp['center']
        anomaly_score = comp.get('anomaly_score', 0)
        anomaly_reasons = comp.get('anomaly_reasons', [])
        
        # 构建显示文本
        score_text = f"Score:{anomaly_score}"
        reasons_text = self._simplify_reasons(anomaly_reasons)
        details_text = (
            f"Area:{comp['area']}, Den:{comp['bbox_density']:.2f}, "
            f"Ar:{comp['aspect_ratio']:.2f}"
        )
        
        text_lines = [score_text, reasons_text, details_text]
        self._draw_text_with_background(image, text_lines, center_x, center_y, color)
    
    def _add_normal_component_labels(self, image: np.ndarray, comp: Dict[str, Any], 
                                   color: Tuple[int, int, int]) -> None:
        """为正常催化剂添加简化标签信息"""
        center_x, center_y = comp['center']
        
        # 为正常催化剂显示密度信息，便于后续分析
        density_text = f"Den:{comp['bbox_density']:.3f}"
        
        # 计算文本尺寸
        text_size = cv2.getTextSize(
            density_text, cv2.FONT_HERSHEY_SIMPLEX, 
            self.config.NORMAL_FONT_SCALE, self.config.NORMAL_FONT_THICKNESS
        )[0]
        
        # 背景框坐标（更紧凑）
        bg_x1 = center_x - text_size[0] // 2 - 3
        bg_y1 = center_y - text_size[1] - 5
        bg_x2 = center_x + text_size[0] // 2 + 3
        bg_y2 = center_y - 2
        
        # 绘制半透明背景框
        overlay = image.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), self.config.COLOR_WHITE, -1)
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 1)
        
        # 混合背景（半透明效果）
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        # 绘制文本
        cv2.putText(
            image, density_text, 
            (center_x - text_size[0] // 2, center_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, self.config.NORMAL_FONT_SCALE, 
            color, self.config.NORMAL_FONT_THICKNESS
        )
    
    def _simplify_reasons(self, anomaly_reasons: List[str]) -> str:
        """简化异常原因显示"""
        if not anomaly_reasons:
            return "[NO_REASON]"
        
        simplified_reasons = []
        reason_map = {
            'aspect ratio': 'AR',
            'circular': 'CIR',
            'irregular': 'IRR',
            'thick': 'THK',
            'thin': 'THN',
            'large': 'LRG',
            'small': 'SML'
        }
        
        for reason in anomaly_reasons:
            for key, abbr in reason_map.items():
                if key in reason:
                    if 'extremely' in reason and key == 'curved':
                        simplified_reasons.append('ECUR')
                    elif 'severely' in reason and key == 'curved':
                        simplified_reasons.append('SCUR')
                    elif key == 'curved':
                        simplified_reasons.append('CUR')
                    else:
                        simplified_reasons.append(abbr)
                    break
        
        return f"[{','.join(simplified_reasons)}]"
    
    def _draw_text_with_background(self, image: np.ndarray, text_lines: List[str], 
                                  center_x: int, center_y: int, 
                                  color: Tuple[int, int, int]) -> None:
        """绘制带背景的文本"""
        # 计算文本尺寸
        line_sizes = []
        max_width = 0
        
        for line_text in text_lines:
            line_size = cv2.getTextSize(
                line_text, cv2.FONT_HERSHEY_SIMPLEX, 
                self.config.FONT_SCALE, self.config.FONT_THICKNESS
            )[0]
            line_sizes.append(line_size)
            max_width = max(max_width, line_size[0])
        
        # 计算总高度
        line_height = line_sizes[0][1] if line_sizes else 20
        total_height = (len(text_lines) * line_height + 
                       (len(text_lines) - 1) * self.config.LINE_SPACING)
        
        # 背景框坐标
        bg_x1 = center_x - max_width // 2 - self.config.BACKGROUND_PADDING
        bg_y1 = center_y - total_height - 10
        bg_x2 = center_x + max_width // 2 + self.config.BACKGROUND_PADDING
        bg_y2 = center_y - self.config.BACKGROUND_PADDING
        
        # 绘制背景框
        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), self.config.COLOR_WHITE, -1)
        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 2)
        
        # 绘制文本
        for i, (line_text, line_size) in enumerate(zip(text_lines, line_sizes)):
            line_x = center_x - line_size[0] // 2
            line_y = (center_y - total_height + (i + 1) * line_height + 
                     i * self.config.LINE_SPACING - 12)
            
            cv2.putText(
                image, line_text, (line_x, line_y),
                cv2.FONT_HERSHEY_SIMPLEX, self.config.FONT_SCALE, 
                color, self.config.FONT_THICKNESS
            )
    
    def _draw_false_positive_regions(self, image: np.ndarray, 
                                   false_positive_regions: List[Dict[str, Any]]) -> np.ndarray:
        """绘制误报区域"""
        fp_mask = np.zeros_like(image)
        fp_color = self.config.COLOR_FALSE_POSITIVE
        
        for fp_region in false_positive_regions:
            # 绘制最小外接矩形
            min_rect = cv2.minAreaRect(fp_region['contour'])
            rect_points = cv2.boxPoints(min_rect)
            rect_points = np.intp(rect_points)
            
            cv2.fillPoly(fp_mask, [rect_points], fp_color)
            cv2.drawContours(image, [fp_region['contour']], -1, fp_color, 3)
            cv2.drawContours(image, [rect_points], -1, fp_color, 2)
            
            # 添加标签
            self._add_false_positive_label(image, fp_region, fp_color)
        
        # 半透明叠加
        return cv2.addWeighted(
            image, self.config.VISUALIZATION_ALPHA, 
            fp_mask, self.config.VISUALIZATION_BETA, 0
        )
    
    def _add_false_positive_label(self, image: np.ndarray, fp_region: Dict[str, Any], 
                                color: Tuple[int, int, int]) -> None:
        """添加误报标签"""
        moments = cv2.moments(fp_region['contour'])
        if moments['m00'] == 0:
            return
        
        center_x = int(moments['m10'] / moments['m00'])
        center_y = int(moments['m01'] / moments['m00'])
        
        # 主标签
        fp_text = "FALSE_POSITIVE"
        text_size = cv2.getTextSize(
            fp_text, cv2.FONT_HERSHEY_SIMPLEX, 
            self.config.FALSE_POSITIVE_FONT_SCALE, 
            self.config.FALSE_POSITIVE_FONT_THICKNESS
        )[0]
        
        cv2.rectangle(
            image, 
            (center_x - text_size[0] // 2 - 5, center_y - text_size[1] - 10),
            (center_x + text_size[0] // 2 + 5, center_y - 5),
            self.config.COLOR_WHITE, -1
        )
        cv2.rectangle(
            image,
            (center_x - text_size[0] // 2 - 5, center_y - text_size[1] - 10),
            (center_x + text_size[0] // 2 + 5, center_y - 5),
            color, 2
        )
        cv2.putText(
            image, fp_text, (center_x - text_size[0] // 2, center_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, self.config.FALSE_POSITIVE_FONT_SCALE, 
            color, self.config.FALSE_POSITIVE_FONT_THICKNESS
        )
        
        # 详细信息
        detail_text = f"Area:{fp_region['area']}, Density:{fp_region['density']:.3f}"
        detail_size = cv2.getTextSize(
            detail_text, cv2.FONT_HERSHEY_SIMPLEX, 
            self.config.DETAIL_FONT_SCALE, self.config.DETAIL_FONT_THICKNESS
        )[0]
        
        cv2.rectangle(
            image,
            (center_x - detail_size[0] // 2 - 3, center_y + 5),
            (center_x + detail_size[0] // 2 + 3, center_y + detail_size[1] + 8),
            self.config.COLOR_WHITE, -1
        )
        cv2.putText(
            image, detail_text, 
            (center_x - detail_size[0] // 2, center_y + detail_size[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX, self.config.DETAIL_FONT_SCALE, 
            self.config.COLOR_BLACK, self.config.DETAIL_FONT_THICKNESS
        )


# ============================================================================
# 主检测器类
# ============================================================================

class CatalystDetector:
    """催化剂异物异形检测器主类"""
    
    def __init__(self, config: DetectionConfig = None):
        self.config = config or DetectionConfig()
        self.image_processor = ImageProcessor(self.config)
        self.unet_inference = UNetInference(self.config)
        self.component_analyzer = ComponentAnalyzer(self.config)
        self.curvature_analyzer = CurvatureAnalyzer(self.config)
        self.component_merger = ComponentMerger(self.config)
        self.fp_filter = FalsePositiveFilter(self.config)
        self.anomaly_classifier = AnomalyClassifier(self.config)
        self.visualizer = ResultVisualizer(self.config)
    
    def detect_foreign_objects(self, mask_unet: np.ndarray, original_image: np.ndarray, 
                             mask_eroded: np.ndarray, 
                             args: argparse.Namespace) -> Tuple[Dict[str, List[Dict[str, Any]]], 
                                                              np.ndarray, List[Dict[str, Any]]]:
        """
        异物异形检测核心算法
        
        Args:
            mask_unet: UNet推理结果掩码
            original_image: 原始图像
            mask_eroded: 预处理掩码
            args: 命令行参数
            
        Returns:
            Tuple[classification_result, mask_filtered, false_positive_regions]
        """
        try:
            # 调整UNet掩码到原图尺寸
            mask_unet_resized = cv2.resize(
                mask_unet, (original_image.shape[1], original_image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            
            # 结合预处理掩码
            mask_eroded_inv = 1 - mask_eroded
            mask_combined = mask_unet_resized & mask_eroded_inv
            mask_combined = mask_combined.astype(np.uint8)
            
            # 形态学操作清理掩码
            mask_clean = self.image_processor.apply_morphological_operations(mask_combined)
            
            # 过滤小连通域
            mask_filtered = self.component_analyzer.filter_small_components(
                mask_clean, args.min_component_area
            )
            
            # 连通域分析
            components_info = self.component_analyzer.analyze_connected_components(mask_filtered)
            
            # 误报过滤
            false_positive_regions = []
            if args.enable_false_positive_filter:
                components_info, false_positive_regions = self.fp_filter.filter_false_positives(
                    components_info, args
                )
            
            # 连通域合并
            if args.enable_component_merge:
                components_info = self.component_merger.merge_connected_components(
                    components_info, args.merge_distance, args.merge_angle_threshold
                )
            
            # 异常分类
            classification_result = self.anomaly_classifier.classify_anomalies(
                components_info, original_image.shape, args
            )
            
            return classification_result, mask_filtered, false_positive_regions
            
        except Exception as e:
            raise DetectionError(f"异物检测失败: {str(e)}")
    
    def process_single_image(self, image_path: str, net: torch.nn.Module, 
                           device: torch.device, args: argparse.Namespace, 
                           output_dir: str) -> Tuple[bool, str, Optional[Dict[str, int]]]:
        """
        单图像处理主函数
        
        Args:
            image_path: 图像路径
            net: UNet网络
            device: 计算设备
            args: 命令行参数
            output_dir: 输出目录
            
        Returns:
            Tuple[success, result_path_or_error, stats]
        """
        try:
            # 预处理
            processed_image, mask_eroded = self.image_processor.preprocess_image(image_path)
            
            # UNet推理
            mask_unet = self.unet_inference.inference(net, device, image_path)
            
            # 读取原图
            original_image = cv2.imread(image_path)
            if original_image is None:
                raise ImageProcessingError(f"无法读取原图: {image_path}")
            
            # 异物异形检测
            classification_result, anomaly_mask, false_positive_regions = self.detect_foreign_objects(
                mask_unet, original_image, mask_eroded, args
            )
            
            # 生成可视化结果
            vis_image = self.visualizer.visualize_results(
                original_image, classification_result, anomaly_mask,
                false_positive_regions, args.show_false_positive, args.show_normal_density
            )
            
            # 保存结果
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_yiwu_result{ext}")
            cv2.imwrite(output_path, vis_image)
            
            # 统计信息
            stats = {
                'foreign_objects_count': len(classification_result['foreign_objects']),
                'deformed_catalysts_count': len(classification_result['deformed_catalysts']),
                'normal_count': len(classification_result['normal']),
                'total_components': sum(len(components) for components in classification_result.values())
            }
            
            logger.info(f"成功处理图像: {image_path}")
            return True, output_path, stats
            
        except Exception as e:
            logger.error(f"处理图像失败 {image_path}: {str(e)}")
            return False, str(e), None


# ============================================================================
# 工具函数
# ============================================================================

def get_image_files(input_dir: str, extensions: List[str]) -> List[str]:
    """
    获取指定目录下的所有图像文件
    
    Args:
        input_dir: 输入目录
        extensions: 支持的扩展名列表
        
    Returns:
        图像文件路径列表
    """
    image_files = []
    for ext in extensions:
        pattern = os.path.join(input_dir, f"*.{ext}")
        image_files.extend(glob.glob(pattern, recursive=False))
        pattern = os.path.join(input_dir, f"*.{ext.upper()}")
        image_files.extend(glob.glob(pattern, recursive=False))
    return sorted(list(set(image_files)))


def load_unet_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """
    加载UNet模型
    
    Args:
        model_path: 模型文件路径
        device: 计算设备
        
    Returns:
        加载的模型
        
    Raises:
        ModelInferenceError: 模型加载失败时抛出
    """
    try:
        net = UNet(n_channels=3, n_classes=2, bilinear=False)
        net.to(device=device)
        
        state_dict = torch.load(model_path, map_location=device)
        _ = state_dict.pop('mask_values', [0, 1])
        net.load_state_dict(state_dict)
        
        logger.info(f'模型加载成功! 使用设备: {device}')
        return net
        
    except Exception as e:
        raise ModelInferenceError(f"模型加载失败: {str(e)}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="催化剂异物异形检测")
    
    # 必需参数
    parser.add_argument('model', type=str, help="UNet模型检查点文件(.pth)")
    
    # 输入输出参数
    parser.add_argument('--input-dir', default='./data/catalyst_merge/origin_data', 
                       type=str, help="输入图像目录")
    parser.add_argument('--output-dir', default='./output/yiwu_results', 
                       type=str, help="输出结果目录")
    parser.add_argument('--image-exts', default=['jpg', 'jpeg', 'png', 'bmp', 'tiff'], 
                       nargs='+', help="支持的图像文件扩展名")
    
    # 异物检测参数
    parser.add_argument('--min-component-area', default=100, type=int, 
                       help="连通域预过滤最小面积阈值")
    parser.add_argument('--min-area', default=500, type=int, help="最小连通域面积阈值")
    parser.add_argument('--max-area', default=50000, type=int, help="最大连通域面积阈值")
    parser.add_argument('--min-aspect-ratio', default=1.5, type=float, help="最小长宽比阈值")
    parser.add_argument('--max-aspect-ratio', default=20.0, type=float, help="最大长宽比阈值")
    parser.add_argument('--min-solidity', default=0.6, type=float, help="最小实心度阈值")
    parser.add_argument('--edge-threshold', default=50, type=int, help="边缘区域阈值(像素)")
    
    # 连通域合并参数
    parser.add_argument('--merge-distance', default=20, type=int, help="连通域合并距离阈值")
    parser.add_argument('--merge-angle-threshold', default=30, type=float, 
                       help="连通域合并角度阈值(度)")
    parser.add_argument('--enable-component-merge', action='store_true', 
                       help="启用智能连通域合并")
    
    # 智能误报过滤参数
    parser.add_argument('--enable-false-positive-filter', action='store_true', default=True,
                       help="启用智能误报过滤算法（默认启用）")
    parser.add_argument('--fp-density-threshold', default=0.4, type=float,
                       help="误报判断密度阈值（越小越严格）")
    parser.add_argument('--fp-area-threshold', default=150000, type=int,
                       help="误报判断面积阈值（绝对值，适用于误报大区域）")
    parser.add_argument('--fp-score-threshold', default=3, type=int,
                       help="误报判断综合评分阈值（越小越严格）")
    parser.add_argument('--show-false-positive', action='store_true', default=False,
                       help="显示误报区域：启用时在结果图中以半透明mask显示检测到的误报区域")
    parser.add_argument('--show-normal-density', action='store_true', default=False,
                       help="显示正常催化剂密度信息：启用时在结果图中显示正常催化剂的bbox_density值")
    
    return parser.parse_args()


def print_summary_statistics(total_stats: Dict[str, int], successful: int, 
                           failed: int, total_images: int, output_dir: str, 
                           args: argparse.Namespace) -> None:
    """打印处理结果统计"""
    print(f"\n{'='*60}")
    print("异物异形检测完成!")
    print(f"{'='*60}")
    print(f"总图像数量: {total_images}")
    print(f"处理成功: {successful}")
    print(f"处理失败: {failed}")
    print(f"结果保存至: {output_dir}")
    
    print("\n检测统计:")
    print(f"  总异物数量: {total_stats['foreign_objects_count']}")
    print(f"  总异形数量: {total_stats['deformed_catalysts_count']}")
    print(f"  总正常数量: {total_stats['normal_count']}")
    if successful > 0:
        avg_components = total_stats['total_components'] / successful
        print(f"  平均每图检测组件数: {avg_components:.1f}")
    
    print("\n检测参数:")
    print(f"  连通域预过滤最小面积: {args.min_component_area}")
    print(f"  最小面积阈值: {args.min_area}")
    print(f"  最大面积阈值: {args.max_area}")
    print(f"  长宽比范围: {args.min_aspect_ratio} - {args.max_aspect_ratio}")
    print(f"  最小实心度: {args.min_solidity}")
    print(f"  边缘阈值: {args.edge_threshold}")
    print(f"  智能合并: {'启用' if args.enable_component_merge else '禁用'}")
    
    if args.enable_component_merge:
        print(f"  合并距离阈值: {args.merge_distance}")
        print(f"  合并角度阈值: {args.merge_angle_threshold}度")
    
    print(f"  智能误报过滤: {'启用' if args.enable_false_positive_filter else '禁用'}")
    if args.enable_false_positive_filter:
        print(f"  误报处理模式: 直接去除")
        print(f"  误报密度阈值: {args.fp_density_threshold}")
        print(f"  误报面积阈值: {args.fp_area_threshold}")
        print(f"  误报评分阈值: {args.fp_score_threshold}")
        print(f"  误报区域可视化: {'启用' if args.show_false_positive else '禁用'}")
        print(f"  正常催化剂密度显示: {'启用' if args.show_normal_density else '禁用'}")


# ============================================================================
# 主函数
# ============================================================================

def main() -> None:
    """主函数"""
    try:
        # 解析参数
        args = parse_args()
        
        # 检查输入目录
        if not os.path.exists(args.input_dir):
            logger.error(f"输入目录 '{args.input_dir}' 不存在!")
            return
        
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 获取图像文件列表
        image_files = get_image_files(args.input_dir, args.image_exts)
        if not image_files:
            logger.error(
                f"在 '{args.input_dir}' 中未找到图像文件，"
                f"支持的扩展名: {args.image_exts}"
            )
            return
        
        logger.info(f"找到 {len(image_files)} 张图像待处理")
        
        # 加载UNet模型
        logger.info("正在加载UNet模型...")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = load_unet_model(args.model, device)
        
        # 初始化检测器
        config = DetectionConfig()
        detector = CatalystDetector(config)
        
        # 处理图像
        successful = 0
        failed = 0
        failed_files = []
        total_stats = {
            'foreign_objects_count': 0,
            'deformed_catalysts_count': 0,
            'normal_count': 0,
            'total_components': 0
        }
        
        logger.info("开始处理图像...")
        for image_path in tqdm(image_files, desc="异物异形检测", unit="图像"):
            success, result, stats = detector.process_single_image(
                image_path, net, device, args, args.output_dir
            )
            
            if success:
                successful += 1
                if stats:
                    for key in total_stats:
                        total_stats[key] += stats[key]
                print(f"✓ 处理完成: {os.path.basename(image_path)} -> {os.path.basename(result)}")
            else:
                failed += 1
                failed_files.append((image_path, result))
                print(f"✗ 处理失败: {image_path}: {result}")
        
        # 输出统计结果
        print_summary_statistics(
            total_stats, successful, failed, len(image_files), 
            args.output_dir, args
        )
        
        if failed_files:
            print("\n失败文件列表:")
            for file_path, error in failed_files:
                print(f"  - {file_path}: {error}")
                
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        raise


if __name__ == '__main__':
    main()