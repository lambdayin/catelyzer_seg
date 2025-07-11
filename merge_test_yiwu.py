import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import glob
import math

import torch
from utils.data_loading import SelfDataset
from unet import UNet


def parse_args():
    parser = argparse.ArgumentParser(description="催化剂异物异形检测")
    parser.add_argument('model', type=str, help="UNet模型检查点文件(.pth)")
    parser.add_argument('--input-dir', default='./data/catalyst_merge/origin_data', type=str, 
                       help="输入图像目录")
    parser.add_argument('--output-dir', default='./output/yiwu_results', type=str, 
                       help="输出结果目录")
    parser.add_argument('--image-exts', default=['jpg', 'jpeg', 'png', 'bmp', 'tiff'], nargs='+', 
                       help="支持的图像文件扩展名")
    
    # 异物检测参数
    parser.add_argument('--min-component-area', default=100, type=int, help="连通域预过滤最小面积阈值")
    parser.add_argument('--min-area', default=500, type=int, help="最小连通域面积阈值")
    parser.add_argument('--max-area', default=50000, type=int, help="最大连通域面积阈值")
    parser.add_argument('--min-aspect-ratio', default=1.5, type=float, help="最小长宽比阈值")
    parser.add_argument('--max-aspect-ratio', default=20.0, type=float, help="最大长宽比阈值")
    parser.add_argument('--min-solidity', default=0.6, type=float, help="最小实心度阈值")
    parser.add_argument('--edge-threshold', default=50, type=int, help="边缘区域阈值(像素)")
    
    # 连通域合并参数
    parser.add_argument('--merge-distance', default=20, type=int, help="连通域合并距离阈值")
    parser.add_argument('--merge-angle-threshold', default=30, type=float, help="连通域合并角度阈值(度)")
    parser.add_argument('--enable-component-merge', action='store_true', help="启用智能连通域合并")
    
    # 🚀 智能误报过滤参数
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
    

    
    return parser.parse_args()


def get_image_files(input_dir, extensions):
    """获取指定目录下的所有图像文件"""
    image_files = []
    for ext in extensions:
        pattern = os.path.join(input_dir, f"*.{ext}")
        image_files.extend(glob.glob(pattern, recursive=False))
        pattern = os.path.join(input_dir, f"*.{ext.upper()}")
        image_files.extend(glob.glob(pattern, recursive=False))
    return sorted(list(set(image_files)))


def pre_process_single_image(image_path):
    """
    图像预处理：处理暗区域
    复用原有的预处理逻辑
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # YUV色彩空间转换，过滤低亮度区域
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    mask_filter = yuv[:, :, 0] < 15
    mask_filter = mask_filter.astype(np.uint8)
    
    # 形态学操作清理掩码
    mask_eroded = cv2.dilate(mask_filter, np.ones((5, 5), np.uint8), iterations=2)
    mask_eroded = cv2.erode(mask_eroded, np.ones((5, 5), np.uint8), iterations=2)
    
    # 将暗区域设置为白色
    image[mask_eroded == 1] = [255, 255, 255]
    return image, mask_eroded


def inference_unet_batch(net, device, image_path):
    """
    UNet模型推理
    复用原有的推理逻辑
    """
    scale_factor = 0.5
    img_ori = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img_ori is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    net.eval()
    img = torch.from_numpy(SelfDataset.preprocess(None, img_ori, scale_factor, is_mask=False))    
    img = img.unsqueeze(0)           
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()        
        mask = output.argmax(dim=1)
        output = (mask[0] * 255).squeeze().numpy().astype(np.uint8)
    return output


def filter_small_components(mask, min_area):
    """
    过滤掉面积过小的连通域
    用于在连通域分析前预处理，去除明显的噪声和小的误检区域
    """
    # 连通域标记
    num_labels, labeled_mask = cv2.connectedComponents(mask)
    
    # 创建过滤后的掩码
    filtered_mask = np.zeros_like(mask)
    
    # 遍历每个连通域（跳过背景label=0）
    for label in range(1, num_labels):
        # 创建当前连通域的掩码
        component_mask = (labeled_mask == label).astype(np.uint8)
        
        # 计算连通域面积
        area = cv2.countNonZero(component_mask)
        
        # 保留面积大于阈值的连通域
        if area >= min_area:
            filtered_mask[component_mask > 0] = 255
    
    return filtered_mask


def calculate_curvature_features(contour, component_mask):
    """
    🌟 优化算法：简化弯曲度分析
    
    通过核心特征精准区分弯曲催化剂和直条状催化剂：
    1. 直线度比例 - 端点直线距离与轮廓长度比值（权重100%）
    2. 骨架线弯曲度 - 基于形态学骨架的弯曲程度（暂时禁用，保留计算）
    """
    
    # 确保轮廓有足够的点
    if len(contour) < 10:
        return {
            'skeleton_curvature': 0,
            'straightness_ratio': 1.0
        }
    
    # 简化轮廓，减少噪声影响
    epsilon = 0.005 * cv2.arcLength(contour, True)
    simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(simplified_contour) < 5:
        simplified_contour = contour
    
    # 1. 🔥 骨架线弯曲度分析（暂时禁用，保留计算）
    skeleton_curvature = calculate_skeleton_curvature(component_mask)
    
    # 2. 🔥 直线度比例分析（权重100%）
    straightness_ratio = calculate_straightness_ratio(simplified_contour)
    
    return {
        'skeleton_curvature': skeleton_curvature,
        'straightness_ratio': straightness_ratio
    }


def calculate_skeleton_curvature(component_mask):
    """
    🔥 核心创新：骨架线弯曲度分析
    提取催化剂的中轴骨架线，分析其弯曲程度
    """
    try:
        # 检查是否有ximgproc模块
        if hasattr(cv2, 'ximgproc'):
            # 形态学骨架提取
            skeleton = cv2.ximgproc.thinning(component_mask)
        else:
            # 如果没有ximgproc，直接使用简化算法
            return calculate_simplified_skeleton_curvature(component_mask)
        
        # 找到骨架线的关键点
        skeleton_points = np.column_stack(np.where(skeleton > 0))
        
        if len(skeleton_points) < 10:
            return 0
        
        # 按空间顺序排列骨架点（简化版路径追踪）
        ordered_points = order_skeleton_points(skeleton_points)
        
        if len(ordered_points) < 5:
            return 0
        
        # 计算骨架线的曲率
        total_curvature = 0
        for i in range(1, len(ordered_points) - 1):
            p1 = ordered_points[i-1]
            p2 = ordered_points[i] 
            p3 = ordered_points[i+1]
            
            # 计算三点间的曲率（使用三角形面积法）
            curvature = calculate_point_curvature(p1, p2, p3)
            total_curvature += curvature
        
        # 归一化：除以骨架长度
        skeleton_length = len(ordered_points)
        return total_curvature / skeleton_length if skeleton_length > 0 else 0
        
    except:
        # 如果没有ximgproc，使用简化的骨架算法
        return calculate_simplified_skeleton_curvature(component_mask)


def calculate_simplified_skeleton_curvature(component_mask):
    """
    简化版骨架弯曲度（不依赖ximgproc）
    使用距离变换+主成分分析评估弯曲程度
    """
    try:
        # 距离变换
        dist_transform = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)
        
        # 找到所有非零像素点
        nonzero_points = np.column_stack(np.where(component_mask > 0))
        
        if len(nonzero_points) < 10:
            return 0
        
        # 使用主成分分析计算形状的主方向
        centered_points = nonzero_points - np.mean(nonzero_points, axis=0)
        cov_matrix = np.cov(centered_points.T)
        
        # 计算特征值
        eigenvalues = np.linalg.eigvals(cov_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]  # 降序排列
        
        if eigenvalues[0] > 1e-6:  # 避免除零错误
            # 计算形状的紧致度（轴比）
            axis_ratio = eigenvalues[1] / eigenvalues[0]
            
            # 计算轮廓的简化弯曲度
            # 轴比越大，形状越接近圆形（可能更弯曲）
            # 但对于细长形状，还需要考虑实际的弯曲程度
            
            # 使用凸包面积比例作为补充
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                contour = contours[0]
                contour_area = cv2.contourArea(contour)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                
                if hull_area > 0:
                    solidity = contour_area / hull_area
                    # 结合轴比和实心度计算弯曲度
                    curvature = axis_ratio * (1 - solidity) * 50  # 调整系数
                    return curvature
            
            return axis_ratio * 20  # 基础评分
            
        return 0
        
    except:
        return 0


def order_skeleton_points(skeleton_points):
    """
    对骨架点进行空间排序，构建连续路径
    简化版：按主方向排序
    """
    if len(skeleton_points) < 3:
        return skeleton_points
    
    # 找到最远的两个点作为端点
    max_dist = 0
    start_idx, end_idx = 0, 0
    
    for i in range(len(skeleton_points)):
        for j in range(i+1, len(skeleton_points)):
            dist = np.linalg.norm(skeleton_points[i] - skeleton_points[j])
            if dist > max_dist:
                max_dist = dist
                start_idx, end_idx = i, j
    
    start_point = skeleton_points[start_idx]
    end_point = skeleton_points[end_idx]
    
    # 沿着从start到end的方向排序点
    direction = end_point - start_point
    direction_norm = np.linalg.norm(direction)
    
    if direction_norm == 0:
        return skeleton_points
    
    direction = direction / direction_norm
    
    # 计算每个点在主方向上的投影
    projections = []
    for point in skeleton_points:
        proj = np.dot(point - start_point, direction)
        projections.append(proj)
    
    # 按投影值排序
    sorted_indices = np.argsort(projections)
    return skeleton_points[sorted_indices]


def calculate_point_curvature(p1, p2, p3):
    """
    计算三个点构成的曲率
    使用三角形面积法计算曲率
    """
    # 向量
    v1 = p2 - p1
    v2 = p3 - p2
    
    # 边长
    d1 = np.linalg.norm(v1)
    d2 = np.linalg.norm(v2)
    
    if d1 == 0 or d2 == 0:
        return 0
    
    # 叉积计算三角形面积
    cross_product = np.cross(v1, v2)
    area = abs(cross_product) / 2.0
    
    # 曲率 = 4 * 面积 / (边长乘积)
    curvature = 4 * area / (d1 * d2 * np.linalg.norm(p3 - p1))
    
    return curvature



def calculate_straightness_ratio(contour):
    """
    🔥 直线度比例分析
    端点距离与轮廓长度的比值
    """
    if len(contour) < 3:
        return 1.0
    
    points = contour.reshape(-1, 2)
    
    # 端点距离
    start_point = points[0]
    end_point = points[-1]
    straight_distance = np.linalg.norm(end_point - start_point)
    
    # 轮廓长度
    contour_length = cv2.arcLength(contour, False)
    
    if contour_length > 0:
        ratio = straight_distance / contour_length
        return ratio
    
    return 1.0


def calculate_component_orientation(contour):
    """
    计算连通域的主方向角度
    """
    if len(contour) < 5:
        return 0
    
    # 使用最小外接矩形的角度
    min_rect = cv2.minAreaRect(contour)
    angle = min_rect[2]
    
    # 标准化角度到-45到45度之间
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    
    return angle


def should_merge_components(comp1, comp2, merge_distance, angle_threshold):
    """
    判断两个连通域是否应该合并
    基于距离、角度和形状相似性
    """
    # 计算中心点距离
    center1 = comp1['center']
    center2 = comp2['center']
    distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    if distance > merge_distance:
        return False
    
    # 计算方向角度差
    angle1 = calculate_component_orientation(comp1['contour'])
    angle2 = calculate_component_orientation(comp2['contour'])
    angle_diff = abs(angle1 - angle2)
    angle_diff = min(angle_diff, 180 - angle_diff)  # 处理角度环形差值
    
    if angle_diff > angle_threshold:
        return False
    
    # 检查面积比例（避免合并过大差异的组件）
    area1, area2 = comp1['area'], comp2['area']
    area_ratio = max(area1, area2) / min(area1, area2)
    
    if area_ratio > 5:  # 面积差异过大
        return False
    
    return True


def merge_connected_components(components_info, merge_distance, angle_threshold):
    """
    智能合并连通域
    合并可能属于同一催化剂的分离连通域
    """
    if not components_info:
        return components_info
    
    # 使用并查集进行连通域合并
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
        for j in range(i+1, n):
            if should_merge_components(components_info[i], components_info[j], 
                                     merge_distance, angle_threshold):
                union(i, j)
    
    # 按合并组重新组织连通域
    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)
    
    # 合并每个组的连通域
    merged_components = []
    for group_indices in groups.values():
        if len(group_indices) == 1:
            # 单个连通域，直接添加
            merged_components.append(components_info[group_indices[0]])
        else:
            # 多个连通域需要合并
            merged_component = merge_component_group(
                [components_info[i] for i in group_indices]
            )
            merged_components.append(merged_component)
    
    return merged_components


def merge_component_group(component_group):
    """
    合并一组连通域为单个连通域
    """
    if len(component_group) == 1:
        return component_group[0]
    
    # 合并所有轮廓点
    all_points = []
    total_area = 0
    
    for comp in component_group:
        all_points.extend(comp['contour'].reshape(-1, 2))
        total_area += comp['area']
    
    # 重新计算凸包
    all_points = np.array(all_points)
    hull = cv2.convexHull(all_points.reshape(-1, 1, 2))
    
    # 重新计算特征
    area = total_area
    min_rect = cv2.minAreaRect(hull)
    width, height = min_rect[1]
    
    if width == 0 or height == 0:
        aspect_ratio = 1.0
    else:
        aspect_ratio = max(width, height) / min(width, height)
    
    # 计算边界框
    x, y, w, h = cv2.boundingRect(hull)
    
    # 计算实心度
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    # 计算圆形度
    perimeter = cv2.arcLength(hull, True)
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    
    # 计算中心点
    moments = cv2.moments(hull)
    if moments['m00'] != 0:
        center_x = int(moments['m10'] / moments['m00'])
        center_y = int(moments['m01'] / moments['m00'])
    else:
        center_x, center_y = x + w//2, y + h//2
    
    # 创建合并后的mask
    merged_mask = np.zeros_like(component_group[0]['mask'])
    for comp in component_group:
        merged_mask = cv2.bitwise_or(merged_mask, comp['mask'])
    
    # 合并anomaly_score和reasons（使用最高分数）
    max_anomaly_score = 0
    combined_reasons = []
    for comp in component_group:
        if 'anomaly_score' in comp:
            max_anomaly_score = max(max_anomaly_score, comp['anomaly_score'])
        if 'anomaly_reasons' in comp:
            combined_reasons.extend(comp['anomaly_reasons'])
    
    # 去重reasons
    unique_reasons = list(set(combined_reasons))
    
    # 🚀 计算合并后的密度特征
    # 1. 最小外接矩形密度
    min_rect_area = width * height
    bbox_density = area / min_rect_area if min_rect_area > 0 else 0
    
    # 2. 轮廓复杂度分析
    hull_indices = cv2.convexHull(hull, returnPoints=False)
    complexity_score = 0
    if len(hull_indices) > 3 and len(hull) > 3:
        try:
            defects = cv2.convexityDefects(hull, hull_indices)
            complexity_score = len(defects) if defects is not None else 0
        except:
            complexity_score = 0
    
    
    return {
        'label': component_group[0]['label'],  # 使用第一个组件的标签
        'area': area,
        'aspect_ratio': aspect_ratio,
        'solidity': solidity,
        'circularity': circularity,
        'center': (center_x, center_y),
        'bbox': (x, y, w, h),
        'min_rect': min_rect,
        'contour': hull,
        'mask': merged_mask,
        'anomaly_score': max_anomaly_score,  # 使用最高的异常分数
        'anomaly_reasons': unique_reasons,    # 合并所有异常原因
        # 🚀 新增密度特征
        'bbox_density': bbox_density,
        'complexity_score': complexity_score
    }


def analyze_connected_components(mask, args=None):
    """
    连通域分析和特征提取
    返回每个连通域的详细特征信息（包含密度特征和弯曲度分类）
    """
    # 连通域标记
    num_labels, labeled_mask = cv2.connectedComponents(mask)
    
    components_info = []
    
    for label in range(1, num_labels):  # 跳过背景(label=0)
        # 创建当前连通域的掩码
        component_mask = (labeled_mask == label).astype(np.uint8)
        
        # 基本几何特征
        # area = cv2.countNonZero(component_mask)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            continue
            
        contour = contours[0]
        area = cv2.contourArea(contour)
        
        # 最小外接矩形
        min_rect = cv2.minAreaRect(contour)
        width, height = min_rect[1]
        if width == 0 or height == 0:
            continue
            
        # 长宽比
        aspect_ratio = max(width, height) / min(width, height)
        
        # 边界框
        x, y, w, h = cv2.boundingRect(contour)
        
        # 实心度 (凸包面积比)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # 圆形度
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # 🚀 密度特征（合并自calculate_region_density）
        # 1. 最小外接矩形密度（更准确的密度计算）
        min_rect_area = width * height
        bbox_density = area / min_rect_area if min_rect_area > 0 else 0
        
        # 2. 轮廓复杂度分析（凸包缺陷数量）
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        complexity_score = 0
        if len(hull_indices) > 3 and len(contour) > 3:
            try:
                defects = cv2.convexityDefects(contour, hull_indices)
                complexity_score = len(defects) if defects is not None else 0
            except:
                complexity_score = 0
        
        
        # 中心点
        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            center_x = int(moments['m10'] / moments['m00'])
            center_y = int(moments['m01'] / moments['m00'])
        else:
            center_x, center_y = x + w//2, y + h//2
        
        component_info = {
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
            'anomaly_score': 0,  # 初始化异常分数
            'anomaly_reasons': [],  # 初始化异常原因列表
            # 🚀 新增密度特征
            'bbox_density': bbox_density,  # 最小外接矩形密度
            'complexity_score': complexity_score  # 轮廓复杂度评分
        }
        
        components_info.append(component_info)
    
    return components_info


def is_false_positive_region(component_info, args):
    """
    判断连通域是否为UNet误报的大区域
    
    优化后的判断逻辑：
    1. 使用专门的误报面积阈值，而不是基于正常催化剂max_area
    2. 使用最小外接矩形密度，对倾斜催化剂更友好
    3. 去除内部空洞检测，避免误杀有空洞的正常催化剂
    4. 保留轮廓复杂度检测，识别真正不规则的误报区域
    """
    area = component_info['area']
    bbox_density = component_info['bbox_density']
    complexity_score = component_info['complexity_score']
    
    # 使用专门针对误报的判断阈值
    is_oversized = area > args.fp_area_threshold  # 使用专门的误报面积阈值
    is_low_density = bbox_density < args.fp_density_threshold  # 最小外接矩形密度过低
    is_complex = complexity_score > 20  # 轮廓过于复杂
    
    # 综合判断逻辑
    false_positive_score = 0
    if is_oversized:
        false_positive_score += 1  # 面积超过误报阈值
    if is_low_density:
        false_positive_score += 3  # 密度过低
    if is_complex:
        false_positive_score += 2  # 复杂度
    
    return false_positive_score >= args.fp_score_threshold  # 使用参数化阈值


def intelligent_component_filtering(components_info, args):
    """
    智能连通域过滤：识别并直接去除UNet误报的大区域
    
    核心算法功能：
    1. 识别UNet误报的大区域
    2. 直接去除误报区域（固定remove模式）
    3. 保留正常尺寸的连通域
    4. 返回误报区域信息（用于可视化）
    """
    filtered_components = []
    false_positive_regions = []
    removed_count = 0
    
    print(f"\n开始智能连通域过滤，初始连通域数量: {len(components_info)}")
    print(f"误报处理模式: 直接去除")
    print(f"误报区域可视化: {'启用' if args.show_false_positive else '禁用'}")
    
    for comp in components_info:
        # 判断是否为误报大区域
        if is_false_positive_region(comp, args):
            print(f"🚫 检测到误报大区域: 面积={comp['area']}, 最小外接矩形密度={comp['bbox_density']:.3f}, 轮廓复杂度={comp['complexity_score']}")
            
            # 保存误报区域信息（用于可视化）
            false_positive_regions.append({
                'mask': comp['mask'],
                'contour': comp['contour'],
                'area': comp['area'],
                'density': comp['bbox_density'],
                'complexity': comp['complexity_score']
            })
            
            # 直接去除误报区域
            removed_count += 1
            print(f"  ❌ 直接去除该误报区域")
            
        else:
            # 保留正常连通域
            filtered_components.append(comp)
    
    print(f"智能过滤完成: 保留正常组件 {len(filtered_components)} 个，去除误报区域 {removed_count} 个")
    print(f"最终连通域数量: {len(filtered_components)}")
    
    return filtered_components, false_positive_regions


def classify_anomalies(components_info, image_shape, args):
    """
    优化的异常区域分类：区分正常催化剂、异物、异形
    采用更宽松的判断条件减少误报
    """
    height, width = image_shape[:2]
    normal_components = []
    foreign_objects = []  # 异物
    deformed_catalysts = []  # 异形催化剂
    
    # 计算图像边缘区域
    edge_threshold = args.edge_threshold
    
    # 🚀 针对当前图片：统计当前图片内所有连通域的最小外接矩形短边分布
    short_sides = []
    for comp in components_info:
        min_rect = comp['min_rect']
        width_rect, height_rect = min_rect[1]
        short_side = min(width_rect, height_rect)
        short_sides.append(short_side)
    
    # 计算当前图片的短边分布统计
    if len(short_sides) >= 3:  # 至少需要3个样本才能计算统计量
        short_sides_array = np.array(short_sides)
        median_short_side = np.median(short_sides_array)
        q75 = np.percentile(short_sides_array, 75)
        q25 = np.percentile(short_sides_array, 25)
        iqr = q75 - q25
        
        # 使用IQR方法定义离群值阈值（针对当前图片）
        if iqr > 0:  # 确保IQR大于0
            outlier_threshold_high = q75 + 1.5 * iqr  # 上界：过粗
            outlier_threshold_low = max(float(q25 - 1.5 * iqr), float(median_short_side * 0.3))  # 下界：过细，但不能太小
        else:
            # IQR为0，所有值相近，使用更宽松的阈值
            outlier_threshold_high = median_short_side * 2.0
            outlier_threshold_low = median_short_side * 0.5
        
        print(f"当前图片短边分布统计: 连通域数={len(short_sides)}, 中位数={median_short_side:.1f}, Q25={q25:.1f}, Q75={q75:.1f}, IQR={iqr:.1f}")
        print(f"当前图片离群值阈值: 过细<{outlier_threshold_low:.1f}, 过粗>{outlier_threshold_high:.1f}")
    else:
        # 样本不足，使用保守阈值（基于当前图片的平均值）
        median_short_side = np.mean(short_sides) if short_sides else 10
        outlier_threshold_high = median_short_side * 2.5
        outlier_threshold_low = median_short_side * 0.4
        print(f"当前图片连通域数量较少({len(short_sides)})，使用保守阈值: 过细<{outlier_threshold_low:.1f}, 过粗>{outlier_threshold_high:.1f}")
    
    for comp in components_info:
        anomaly_score = 0
        anomaly_reasons = []
        
        # 1. 尺寸异常检测（使用评分制度而非硬阈值）
        # if comp['area'] < args.min_area * 0.7:  # 更宽松的面积阈值
        #     anomaly_score += 2
        #     anomaly_reasons.append('area is too small')
        # elif comp['area'] > args.max_area * 1.2:  # 更宽松的面积阈值
        #     anomaly_score += 2
        #     anomaly_reasons.append('area is too large')
        # elif comp['area'] < args.min_area:
        #     anomaly_score += 1  # 轻微异常
        #     anomaly_reasons.append('area is slightly small')
        if comp['area'] > args.max_area * 1.2:  # 更宽松的面积阈值
            anomaly_score += 2
            anomaly_reasons.append('area is too large')
        
        # 2. 形状异常检测（更宽松的长宽比）
        if comp['aspect_ratio'] < args.min_aspect_ratio * 0.8:
            anomaly_score += 2
            anomaly_reasons.append('aspect ratio is too small')
        elif comp['aspect_ratio'] > args.max_aspect_ratio * 1.2:
            anomaly_score += 2
            anomaly_reasons.append('aspect ratio is too large')
        elif comp['aspect_ratio'] < args.min_aspect_ratio:
            anomaly_score += 1  # 轻微异常
            anomaly_reasons.append('aspect ratio is slightly small')
        
        # 3. 实心度异常检测（更宽松的实心度）
        if comp['solidity'] < args.min_solidity * 0.8:  # 更宽松
            anomaly_score += 2
            anomaly_reasons.append('shape is irregular')
        elif comp['solidity'] < args.min_solidity:
            anomaly_score += 1  # 轻微异常
            anomaly_reasons.append('shape is slightly irregular')
        
        # 4. 圆形度异常检测（更严格的圆形度阈值）
        if comp['circularity'] > 0.8:  # 提高阈值，减少误报
            anomaly_score += 2
            anomaly_reasons.append('shape is too circular')
        elif comp['circularity'] > 0.7:
            anomaly_score += 1  # 轻微异常
            anomaly_reasons.append('shape is slightly circular')
        
        # 🚀 5. 新增：短边离群检测（基于当前图片分布检测过粗或过细的催化剂）
        min_rect = comp['min_rect']
        width_rect, height_rect = min_rect[1]
        component_short_side = min(width_rect, height_rect)
        
        if component_short_side > 2 * outlier_threshold_high:
            # 短边过长（催化剂过粗），相对于当前图片内其他催化剂明显过粗
            anomaly_score += 3  # 高分数
            anomaly_reasons.append('short side is too thick (outlier)')
            print(f"检测到过粗组件: 短边={component_short_side:.1f} > 当前图片阈值{outlier_threshold_high:.1f}")
        # elif component_short_side < outlier_threshold_low:
        #     # 短边过短（催化剂过细），相对于当前图片内其他催化剂明显过细
        #     anomaly_score += 2  # 中等分数
        #     anomaly_reasons.append('short side is too thin (outlier)')
        #     print(f"检测到过细组件: 短边={component_short_side:.1f} < 当前图片阈值{outlier_threshold_low:.1f}")
        

        
        # 7. 综合评分判断
        comp['anomaly_score'] = anomaly_score
        comp['anomaly_reasons'] = anomaly_reasons
        
        # 使用评分制度进行分类（考虑新增的短边离群检测）
        if anomaly_score <= 1:  # 正常或轻微异常
            normal_components.append(comp)
        elif anomaly_score == 2:  # 中等异常
            # 更保守的分类，倾向于归类为正常
            if (comp['circularity'] > 0.8 or  # 只有非常圆的才认为是异物
                comp['area'] < args.min_area * 0.5 or  # 或者面积极小
                'short side is too thick (outlier)' in anomaly_reasons):  # 或者明显过粗
                foreign_objects.append(comp)
            else:
                normal_components.append(comp)  # 归类为正常
        elif anomaly_score >= 3 and anomaly_score <= 6:  # 明显异常
            # 区分异物和异形催化剂
            if (comp['area'] < args.min_area * 1.5):
                normal_components.append(comp)
            elif (comp['circularity'] > 0.7 or          # 较圆
                'shape is too circular' in anomaly_reasons or
                'short side is too thick (outlier)' in anomaly_reasons):  # 过粗通常是异物
                foreign_objects.append(comp)
            else:
                deformed_catalysts.append(comp)
        else:  # 高异常 (score > 6)
            # 高分数异常，更严格分类
            if (comp['area'] < args.min_area * 1.5 or  # 面积较小
                comp['circularity'] > 0.7 or          # 较圆
                'shape is too circular' in anomaly_reasons or
                'short side is too thick (outlier)' in anomaly_reasons):  # 过粗
                foreign_objects.append(comp)
            else:
                deformed_catalysts.append(comp)
    
    return {
        'normal': normal_components,
        'foreign_objects': foreign_objects,
        'deformed_catalysts': deformed_catalysts
    }


def detect_foreign_objects(mask_unet, original_image, mask_eroded, args):
    """
    异物异形检测核心算法
    """
    # 将UNet掩码调整到原图尺寸
    mask_unet_resized = cv2.resize(mask_unet, (original_image.shape[1], original_image.shape[0]), 
                                  interpolation=cv2.INTER_NEAREST)
    
    # 结合预处理掩码
    mask_eroded_inv = 1 - mask_eroded
    mask_combined = mask_unet_resized & mask_eroded_inv
    mask_combined = mask_combined.astype(np.uint8)
    
    # 形态学操作清理掩码
    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.erode(mask_combined, kernel, iterations=1)
    mask_clean = cv2.dilate(mask_clean, kernel, iterations=1)
    # mask_clean = cv2.dilate(mask_clean, kernel, iterations=1)
    # mask_clean = cv2.erode(mask_clean, kernel, iterations=1)

    # 使用开运算去除小的噪声点
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    # 使用闭运算填充内部的小孔洞
    # mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    
    # 过滤掉面积过小的连通域（去除折叠催化剂的小露出部分等噪声）
    mask_filtered = filter_small_components(mask_clean, args.min_component_area)
    
    # 连通域分析
    components_info = analyze_connected_components(mask_filtered, args)
    
    
    # 识别并处理UNet误报的大区域，从中提取真正的催化剂
    false_positive_regions = []
    if args.enable_false_positive_filter:
        components_info, false_positive_regions = intelligent_component_filtering(components_info, args)
    
    # 智能连通域合并（可选）
    if args.enable_component_merge:
        components_info = merge_connected_components(
            components_info, args.merge_distance, args.merge_angle_threshold)
    
    # 异常分类（现在包含弯曲度异常检测）
    classification_result = classify_anomalies(components_info, original_image.shape, args)
    
    return classification_result, mask_filtered, false_positive_regions


def visualize_results(original_image, classification_result, anomaly_mask, false_positive_regions=None, show_false_positive=False):
    """
    生成可视化结果
    显示整体催化剂连通域mask叠加效果，用不同颜色标注不同类型
    可选显示误报区域的半透明mask
    对异常组件显示anomaly_score评分
    
    显示内容：
    - 第一行：Score:x（异常评分）
    - 第二行：[原因代码]
    """
    vis_image = original_image.copy()
    
    # 创建彩色mask
    colored_mask = np.zeros_like(original_image)
    
    # 颜色定义 - 使用更鲜艳的颜色
    colors = {
        'foreign_objects': (0, 0, 255),      # 鲜红色 - 异物
        'deformed_catalysts': (0, 128, 255), # 鲜橙色 - 异形催化剂  
        'normal': (0, 255, 0),               # 鲜绿色 - 正常催化剂
    }
    
    # 标签文本
    labels = {
        'foreign_objects': 'foreign_objects',
        'deformed_catalysts': 'deformed_catalysts',
        'normal': 'normal',
    }
    
    # 绘制所有连通域的mask
    for category, components in classification_result.items():
        if category not in colors:  # 跳过不需要显示的类别
            continue
            
        color = colors[category]
        label_text = labels[category]
        
        for comp in components:
            # 填充整个连通域区域
            colored_mask[comp['mask'] > 0] = color
            
            # 绘制轮廓边界
            # cv2.drawContours(vis_image, [comp['contour']], -1, color, 3)
            
            # 绘制最小外接矩形
            min_rect = comp['min_rect']
            rect_points = cv2.boxPoints(min_rect)
            rect_points = np.int32(rect_points).reshape((-1, 1, 2))
            cv2.drawContours(vis_image, [rect_points], -1, color, 2)
            
            # 添加标签信息
            center_x, center_y = comp['center']
            
            # 为异常组件（异物和异形催化剂）和密度较低的正常催化剂添加anomaly_score和anomaly_reasons标签
            # if category in ['foreign_objects', 'deformed_catalysts'] or (category == 'normal' and comp['bbox_density'] < 0.6 and comp['complexity_score'] > 10):
            if category in ['foreign_objects', 'deformed_catalysts'] or (category == 'normal' and comp['aspect_ratio'] < 2):
                anomaly_score = comp.get('anomaly_score', 0)
                anomaly_reasons = comp.get('anomaly_reasons', [])
                
                # 构建显示文本
                score_text = f"Score:{anomaly_score}"
                
                # 处理异常原因：简化长文本，只显示关键信息
                if anomaly_reasons:
                    # 简化异常原因的显示
                    simplified_reasons = []
                    for reason in anomaly_reasons:
                        if 'aspect ratio' in reason:
                            simplified_reasons.append('AR')  # Aspect Ratio
                        elif 'circular' in reason:
                            simplified_reasons.append('CIR')  # Circular
                        elif 'irregular' in reason:
                            simplified_reasons.append('IRR')  # Irregular
                        elif 'thick' in reason:
                            simplified_reasons.append('THK')  # Thick
                        elif 'thin' in reason:
                            simplified_reasons.append('THN')  # Thin
                        elif 'curved' in reason:
                            if 'extremely' in reason:
                                simplified_reasons.append('ECUR')  # Extremely Curved
                            elif 'severely' in reason:
                                simplified_reasons.append('SCUR')  # Severely Curved
                            else:
                                simplified_reasons.append('CUR')   # Curved
                        elif 'large' in reason:
                            simplified_reasons.append('LRG')  # Large
                        elif 'small' in reason:
                            simplified_reasons.append('SML')  # Small
                    
                    reasons_text = f"[{','.join(simplified_reasons)}]"
                else:
                    reasons_text = "[NO_REASON]"
                
                # 第三行：详细信息（面积、密度、复杂度）
                details_text = f"Area:{comp['area']}, Den:{comp['bbox_density']:.2f}, Ar:{comp['aspect_ratio']:.2f}"
                
                # 准备显示的文本行
                text_lines = [score_text, reasons_text, details_text]
                
                # 计算所有文字行的尺寸
                font_scale = 0.6
                font_thickness = 2
                line_sizes = []
                max_width = 0
                
                for line_text in text_lines:
                    line_size = cv2.getTextSize(line_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                    line_sizes.append(line_size)
                    max_width = max(max_width, line_size[0])
                
                # 计算总高度（行间距8像素）
                line_height = line_sizes[0][1] if line_sizes else 20
                total_height = len(text_lines) * line_height + (len(text_lines) - 1) * 8
                
                # 背景框坐标
                bg_x1 = center_x - max_width//2 - 5
                bg_y1 = center_y - total_height - 10
                bg_x2 = center_x + max_width//2 + 5
                bg_y2 = center_y - 5
                
                # 绘制背景框
                cv2.rectangle(vis_image, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)
                cv2.rectangle(vis_image, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 2)
                
                # 逐行添加文字
                for i, (line_text, line_size) in enumerate(zip(text_lines, line_sizes)):
                    line_x = center_x - line_size[0]//2
                    line_y = center_y - total_height + (i + 1) * line_height + i * 8 - 12
                    
                    # 使用默认颜色
                    text_color = color
                    
                    cv2.putText(vis_image, line_text, (line_x, line_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
    
    # 绘制误报区域（如果启用且有误报区域）
    if show_false_positive and false_positive_regions:
        # 创建误报区域的mask
        fp_mask = np.zeros_like(original_image)
        fp_color = (128, 0, 128)  # 紫色表示误报区域
        
        for fp_region in false_positive_regions:
            # 计算最小外接矩形
            min_rect = cv2.minAreaRect(fp_region['contour'])
            rect_points = cv2.boxPoints(min_rect)
            rect_points = np.intp(rect_points)
            
            # 使用最小外接矩形的半透明mask显示误报区域
            cv2.fillPoly(fp_mask, [rect_points], fp_color)
            
            # 绘制误报区域的轮廓边界（保持原轮廓）
            cv2.drawContours(vis_image, [fp_region['contour']], -1, fp_color, 3)
            
            # 绘制最小外接矩形边界
            cv2.drawContours(vis_image, [rect_points], -1, fp_color, 2)
            
            # 在误报区域中心添加标签
            moments = cv2.moments(fp_region['contour'])
            if moments['m00'] != 0:
                center_x = int(moments['m10'] / moments['m00'])
                center_y = int(moments['m01'] / moments['m00'])
                
                # 添加误报标签
                fp_text = "FALSE_POSITIVE"
                text_size = cv2.getTextSize(fp_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(vis_image, (center_x-text_size[0]//2-5, center_y-text_size[1]-10), 
                             (center_x+text_size[0]//2+5, center_y-5), (255, 255, 255), -1)
                cv2.rectangle(vis_image, (center_x-text_size[0]//2-5, center_y-text_size[1]-10), 
                             (center_x+text_size[0]//2+5, center_y-5), fp_color, 2)
                cv2.putText(vis_image, fp_text, (center_x-text_size[0]//2, center_y-8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, fp_color, 2)
                
                # 添加详细信息
                detail_text = f"Area:{fp_region['area']}, Density:{fp_region['density']:.3f}"
                detail_size = cv2.getTextSize(detail_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(vis_image, (center_x-detail_size[0]//2-3, center_y+5), 
                             (center_x+detail_size[0]//2+3, center_y+detail_size[1]+8), (255, 255, 255), -1)
                cv2.putText(vis_image, detail_text, (center_x-detail_size[0]//2, center_y+detail_size[1]+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # 将误报mask以半透明形式叠加到图像上
        vis_image = cv2.addWeighted(vis_image, 0.8, fp_mask, 0.2, 0)
    
    # 将彩色mask叠加到原图上
    # cv2.imwrite('vis_image.png', vis_image)
    # mask_overlay = cv2.addWeighted(vis_image, 0.6, colored_mask, 0.4, 0)
    mask_overlay = vis_image
    

    
    final_result = mask_overlay
    
    return final_result


def process_single_image_yiwu(image_path, net, device, args, output_dir):
    """
    单图像异物异形检测处理主函数
    """
    try:
        # 预处理
        processed_image, mask_eroded = pre_process_single_image(image_path)
        
        # UNet推理
        mask_unet = inference_unet_batch(net, device, image_path)
        
        # 读取原图
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"无法读取原图: {image_path}")
        
        # 异物异形检测
        classification_result, anomaly_mask, false_positive_regions = detect_foreign_objects(
            mask_unet, original_image, mask_eroded, args)
        
        # 生成可视化结果
        vis_image = visualize_results(original_image, classification_result, anomaly_mask, 
                                    false_positive_regions, args.show_false_positive)
        
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
        
        return True, output_path, stats
        
    except Exception as e:
        return False, str(e), None


def main():
    args = parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录 '{args.input_dir}' 不存在!")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取图像文件列表
    image_files = get_image_files(args.input_dir, args.image_exts)
    if not image_files:
        print(f"在 '{args.input_dir}' 中未找到图像文件，支持的扩展名: {args.image_exts}")
        return
    
    print(f"找到 {len(image_files)} 张图像待处理")
    
    # 加载UNet模型
    print("正在加载UNet模型...")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    net = UNet(n_channels=3, n_classes=2, bilinear=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    
    try:
        state_dict = torch.load(args.model, map_location=device)
        _ = state_dict.pop('mask_values', [0, 1])
        net.load_state_dict(state_dict)
        print(f'模型加载成功! 使用设备: {device}')
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
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
    
    print("开始处理图像...")
    for image_path in tqdm(image_files, desc="异物异形检测", unit="图像"):
        success, result, stats = process_single_image_yiwu(image_path, net, device, args, args.output_dir)
        
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
    
    # 输出处理结果统计
    print(f"\n{'='*60}")
    print(f"异物异形检测完成!")
    print(f"{'='*60}")
    print(f"总图像数量: {len(image_files)}")
    print(f"处理成功: {successful}")
    print(f"处理失败: {failed}")
    print(f"结果保存至: {args.output_dir}")
    print(f"\n检测统计:")
    print(f"  总异物数量: {total_stats['foreign_objects_count']}")
    print(f"  总异形数量: {total_stats['deformed_catalysts_count']}")
    print(f"  总正常数量: {total_stats['normal_count']}")
    print(f"  平均每图检测组件数: {total_stats['total_components']/successful:.1f}" if successful > 0 else "")
    
    if failed_files:
        print(f"\n失败文件列表:")
        for file_path, error in failed_files:
            print(f"  - {file_path}: {error}")
    
    print(f"\n检测参数:")
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
    print(f"  🚀 智能误报过滤: {'启用' if args.enable_false_positive_filter else '禁用'}")
    if args.enable_false_positive_filter:
        print(f"  误报处理模式: 直接去除")
        print(f"  误报密度阈值: {args.fp_density_threshold}")
        print(f"  误报面积阈值: {args.fp_area_threshold}")
        print(f"  误报评分阈值: {args.fp_score_threshold}")
        print(f"  误报区域可视化: {'启用' if args.show_false_positive else '禁用'}")



if __name__ == '__main__':
    main() 