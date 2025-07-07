import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import math

import torch
from utils.data_loading import SelfDataset
from unet import UNet

# Set font for plots
rcParams['font.sans-serif'] = ['DejaVu Sans']  
rcParams['axes.unicode_minus'] = False

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze connected component sub-pixel area distribution")
    parser.add_argument('model', type=str, help="UNet model checkpoint file (.pth)")
    parser.add_argument('--input-dir', default='./test_0527/imgdata', type=str, help="Input image directory")
    parser.add_argument('--output-dir', default='./area_analysis_results', type=str, help="Analysis results output directory")
    parser.add_argument('--image-exts', default=['jpg', 'jpeg', 'png', 'bmp', 'tiff'], nargs='+', 
                       help="Image file extensions to process")
    
    # 连通域过滤参数
    parser.add_argument('--min-component-area', default=200, type=int, help="连通域预过滤最小面积阈值")
    
    # 连通域合并参数
    parser.add_argument('--merge-distance', default=20, type=int, help="连通域合并距离阈值")
    parser.add_argument('--merge-angle-threshold', default=30, type=float, help="连通域合并角度阈值(度)")
    parser.add_argument('--enable-component-merge', action='store_true', default=True, help="启用智能连通域合并")
    
    # 智能误报过滤参数
    parser.add_argument('--enable-false-positive-filter', action='store_true', default=True, 
                       help="启用智能误报过滤算法（默认启用）")
    parser.add_argument('--fp-density-threshold', default=0.4, type=float, 
                       help="误报判断密度阈值（越小越严格）")
    parser.add_argument('--fp-area-threshold', default=5000, type=int,
                       help="误报判断面积阈值（绝对值，适用于误报大区域）")
    parser.add_argument('--fp-score-threshold', default=4, type=int, 
                       help="误报判断综合评分阈值（越小越严格）")
    
    return parser.parse_args()

def get_image_files(input_dir, extensions):
    """Get all image files"""
    image_files = []
    for ext in extensions:
        pattern = os.path.join(input_dir, f"*.{ext}")
        image_files.extend(glob.glob(pattern, recursive=False))
        pattern = os.path.join(input_dir, f"*.{ext.upper()}")
        image_files.extend(glob.glob(pattern, recursive=False))
    return sorted(list(set(image_files)))

def pre_process_single_image(image_path):
    """Preprocess single image, consistent with merge_test.py"""
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

def inference_unet_batch(net, device, image_path):
    """UNet inference, consistent with merge_test.py"""
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


def analyze_connected_components_advanced(mask):
    """
    连通域分析和特征提取
    返回每个连通域的详细特征信息
    """
    # 连通域标记
    num_labels, labeled_mask = cv2.connectedComponents(mask)
    
    components_info = []
    
    for label in range(1, num_labels):  # 跳过背景(label=0)
        # 创建当前连通域的掩码
        component_mask = (labeled_mask == label).astype(np.uint8)
        
        # 基本几何特征
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
            'mask': component_mask
        }
        
        components_info.append(component_info)
    
    return components_info


def calculate_region_density(component_mask):
    """
    计算连通域的密度特征
    """
    # 1. 最小外接矩形密度（更准确的密度计算）
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        contour = contours[0]
        actual_area = cv2.contourArea(contour)
        # 使用最小外接矩形而不是正外接矩形，对倾斜物体更准确
        min_rect = cv2.minAreaRect(contour)
        min_rect_area = min_rect[1][0] * min_rect[1][1]  # width * height
        bbox_density = actual_area / min_rect_area if min_rect_area > 0 else 0
    else:
        bbox_density = 0
    
    # 2. 轮廓复杂度分析
    if len(contours) > 0:
        contour = contours[0]
        # 计算轮廓的convex defects
        hull = cv2.convexHull(contour, returnPoints=False)
        if len(hull) > 3 and len(contour) > 3:
            try:
                defects = cv2.convexityDefects(contour, hull)
                complexity_score = len(defects) if defects is not None else 0
            except:
                complexity_score = 0
        else:
            complexity_score = 0
    else:
        complexity_score = 0
    
    return {
        'bbox_density': bbox_density,
        'complexity_score': complexity_score
    }


def is_false_positive_region(component_info, density_info, args):
    """
    判断连通域是否为UNet误报的大区域
    """
    area = component_info['area']
    bbox_density = density_info['bbox_density']
    complexity_score = density_info['complexity_score']
    
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
    智能连通域过滤：识别并处理UNet误报的大区域
    """
    filtered_components = []
    extracted_components = []
    false_positive_regions = []
    removed_count = 0
    
    for comp in components_info:
        # 计算密度特征
        density_info = calculate_region_density(comp['mask'])
        
        # 判断是否为误报大区域
        if is_false_positive_region(comp, density_info, args):
            # 保存误报区域信息
            false_positive_regions.append({
                'mask': comp['mask'],
                'contour': comp['contour'],
                'area': comp['area'],
                'density': density_info['bbox_density'],
                'complexity': density_info['complexity_score']
            })

            removed_count += 1
        else:
            # 保留正常连通域
            filtered_components.append(comp)
    
    # 合并过滤后的连通域和提取的组件
    final_components = filtered_components + extracted_components
    
    return final_components, false_positive_regions


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
        'mask': merged_mask
    }


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


def analyze_connected_components_basic(image_path, mask_eroded, mask_unet):
    """Analyze connected components and collect sub-pixel area data (original function)"""
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    mask_eroded = 1 - mask_eroded
    mask_unet = cv2.resize(mask_unet, (orig_img.shape[1], orig_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask_check = mask_unet & mask_eroded
    mask_check = mask_check.astype(np.uint8)
    mask_check = cv2.erode(mask_check, np.ones((3, 3), np.uint8), iterations=2)
    mask_check = cv2.dilate(mask_check, np.ones((3, 3), np.uint8), iterations=2)
    mask_check = cv2.dilate(mask_check, np.ones((3, 3), np.uint8), iterations=1)
    mask_check = cv2.erode(mask_check, np.ones((3, 3), np.uint8), iterations=1)

    # Use opening operation to remove small noise points
    mask_check = cv2.morphologyEx(mask_check, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

    _, labeled_mask = cv2.connectedComponents(mask_check)
    labeled_mask = np.uint8(labeled_mask)

    contours, __ = cv2.findContours(labeled_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    contour_areas = []      # Sub-pixel level connected component area
    bounding_rect_areas = [] # Bounding rectangle area
    perimeters = []         # Perimeter
    aspect_ratios = []      # Aspect ratio
    min_rect_areas = []     # Minimum bounding rectangle area
    
    for contour in contours:
        if len(contour) < 5:  # Skip if connected component is too small
            continue
            
        # Calculate sub-pixel level connected component area
        area = cv2.contourArea(contour)
        contour_areas.append(area)
        
        # Calculate perimeter
        perimeter = cv2.arcLength(contour, True)
        perimeters.append(perimeter)
        
        # Calculate bounding rectangle area
        x, y, w, h = cv2.boundingRect(contour)
        bounding_rect_areas.append(w * h)
        
        # Calculate aspect ratio
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        aspect_ratios.append(aspect_ratio)
        
        # Calculate minimum bounding rectangle area
        min_rect = cv2.minAreaRect(contour)
        min_width, min_height = min_rect[1]
        min_rect_areas.append(min_width * min_height)
    
    return contour_areas, bounding_rect_areas, min_rect_areas, perimeters, aspect_ratios


def process_single_image(image_path, net, device, args):
    """Process single image and return area data with advanced filtering"""
    try:
        # 预处理
        _, mask_eroded = pre_process_single_image(image_path)
        mask_unet = inference_unet_batch(net, device, image_path)
        
        # 读取原图
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # 基础的mask处理（复用原有逻辑）
        mask_eroded_inv = 1 - mask_eroded
        mask_unet_resized = cv2.resize(mask_unet, (original_image.shape[1], original_image.shape[0]), 
                                      interpolation=cv2.INTER_NEAREST)
        mask_combined = mask_unet_resized & mask_eroded_inv
        mask_combined = mask_combined.astype(np.uint8)
        
        # 形态学操作清理掩码
        kernel = np.ones((3, 3), np.uint8)
        mask_clean = cv2.erode(mask_combined, kernel, iterations=2)
        mask_clean = cv2.dilate(mask_clean, kernel, iterations=2)
        mask_clean = cv2.dilate(mask_clean, kernel, iterations=1)
        mask_clean = cv2.erode(mask_clean, kernel, iterations=1)

        # 使用开运算去除小的噪声点
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
        
        # 🚀 新增：过滤掉面积过小的连通域（去除折叠催化剂的小露出部分等噪声）
        mask_filtered = filter_small_components(mask_clean, args.min_component_area)
        
        # 🚀 新增：连通域分析
        components_info = analyze_connected_components_advanced(mask_filtered)
        
        # 🚀 新增：识别并处理UNet误报的大区域
        false_positive_regions = []
        if args.enable_false_positive_filter:
            components_info, false_positive_regions = intelligent_component_filtering(components_info, args)
        
        # 🚀 新增：智能连通域合并（可选）
        if args.enable_component_merge:
            components_info = merge_connected_components(
                components_info, args.merge_distance, args.merge_angle_threshold
            )
        
        # 直接从components_info提取统计数据（无需异常分类）
        contour_areas = [comp['area'] for comp in components_info]
        perimeters = []
        aspect_ratios = []
        bounding_rect_areas = []
        min_rect_areas = []
        solidities = []
        circularities = []
        
        for comp in components_info:
            # 周长
            perimeter = cv2.arcLength(comp['contour'], True)
            perimeters.append(perimeter)
            
            # 长宽比
            aspect_ratios.append(comp['aspect_ratio'])
            
            # 边界框面积
            x, y, w, h = comp['bbox']
            bounding_rect_areas.append(w * h)
            
            # 最小外接矩形面积
            min_rect = comp['min_rect']
            min_width, min_height = min_rect[1]
            min_rect_areas.append(min_width * min_height)
            
            # 实心度和圆形度
            solidities.append(comp['solidity'])
            circularities.append(comp['circularity'])
        
        filename = os.path.basename(image_path)
        return True, {
            'filename': filename,
            'contour_areas': contour_areas,
            'bounding_rect_areas': bounding_rect_areas,
            'min_rect_areas': min_rect_areas,
            'perimeters': perimeters,
            'aspect_ratios': aspect_ratios,
            'solidities': solidities,
            'circularities': circularities,
            'component_count': len(contour_areas),
            'components_info': components_info,  # 保存原始组件信息
            'false_positive_regions_count': len(false_positive_regions)
        }
    except Exception as e:
        return False, str(e)

def create_analysis_plots(all_data, output_dir):
    """Create analysis plots for connected components distribution"""
    # Collect all data
    all_contour_areas = []
    all_bounding_rect_areas = []
    all_min_rect_areas = []
    all_perimeters = []
    all_aspect_ratios = []
    all_solidities = []
    all_circularities = []
    image_stats = []
    
    for data in all_data:
        all_contour_areas.extend(data['contour_areas'])
        all_bounding_rect_areas.extend(data['bounding_rect_areas'])
        all_min_rect_areas.extend(data['min_rect_areas'])
        all_perimeters.extend(data['perimeters'])
        all_aspect_ratios.extend(data['aspect_ratios'])
        all_solidities.extend(data.get('solidities', []))
        all_circularities.extend(data.get('circularities', []))
        
        image_stats.append({
            'filename': data['filename'],
            'component_count': data['component_count'],
            'total_contour_area': sum(data['contour_areas']),
            'avg_contour_area': np.mean(data['contour_areas']) if data['contour_areas'] else 0,
            'max_contour_area': max(data['contour_areas']) if data['contour_areas'] else 0,
            'avg_perimeter': np.mean(data['perimeters']) if data['perimeters'] else 0,
            'avg_aspect_ratio': np.mean(data['aspect_ratios']) if data['aspect_ratios'] else 0,
            'avg_solidity': np.mean(data.get('solidities', [])) if data.get('solidities') else 0,
            'avg_circularity': np.mean(data.get('circularities', [])) if data.get('circularities') else 0,
            'false_positive_regions_count': data.get('false_positive_regions_count', 0)
        })
    
    # Create plots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Connected component area distribution histogram
    plt.subplot(3, 4, 1)
    plt.hist(all_contour_areas, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Connected Component Area (pixels²)')
    plt.ylabel('Frequency')
    plt.title('Connected Component Area Distribution')
    plt.grid(True, alpha=0.3)
    
    # 2. Area distribution box plot
    plt.subplot(3, 4, 2)
    plt.boxplot(all_contour_areas)
    plt.ylabel('Connected Component Area (pixels²)')
    plt.title('Area Distribution Box Plot')
    plt.grid(True, alpha=0.3)
    
    # 3. Perimeter distribution histogram
    plt.subplot(3, 4, 3)
    plt.hist(all_perimeters, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.xlabel('Perimeter (pixels)')
    plt.ylabel('Frequency')
    plt.title('Perimeter Distribution')
    plt.grid(True, alpha=0.3)
    
    # 4. Aspect ratio distribution
    plt.subplot(3, 4, 4)
    plt.hist(all_aspect_ratios, bins=30, alpha=0.7, color='purple', edgecolor='black')
    plt.xlabel('Aspect Ratio')
    plt.ylabel('Frequency')
    plt.title('Aspect Ratio Distribution')
    plt.grid(True, alpha=0.3)
    
    # 5. Number of connected components per image
    plt.subplot(3, 4, 5)
    component_counts = [stat['component_count'] for stat in image_stats]
    plt.bar(range(len(component_counts)), component_counts, alpha=0.7, color='orange')
    plt.xlabel('Image Index')
    plt.ylabel('Number of Connected Components')
    plt.title('Components Count per Image')
    plt.grid(True, alpha=0.3)
    
    # 6. Area vs Perimeter scatter plot
    plt.subplot(3, 4, 6)
    plt.scatter(all_perimeters, all_contour_areas, alpha=0.6, s=30)
    plt.xlabel('Perimeter (pixels)')
    plt.ylabel('Area (pixels²)')
    plt.title('Area vs Perimeter Relationship')
    plt.grid(True, alpha=0.3)
    
    # 7. Log scale area distribution
    plt.subplot(3, 4, 7)
    log_areas = np.log10(np.array(all_contour_areas) + 1e-6)
    plt.hist(log_areas, bins=30, alpha=0.7, color='red', edgecolor='black')
    plt.xlabel('log₁₀(Area) (pixels²)')
    plt.ylabel('Frequency')
    plt.title('Area Distribution (Log Scale)')
    plt.grid(True, alpha=0.3)
    
    # 8. Comparison of different area calculation methods
    plt.subplot(3, 4, 8)
    plt.scatter(all_contour_areas, all_min_rect_areas, alpha=0.6, s=30, label='Min Bounding Rect')
    plt.scatter(all_contour_areas, all_bounding_rect_areas, alpha=0.6, s=30, label='Bounding Rect')
    plt.plot([0, max(all_contour_areas)], [0, max(all_contour_areas)], 'k--', alpha=0.5, label='y=x')
    plt.xlabel('Connected Component Area (pixels²)')
    plt.ylabel('Rectangle Area (pixels²)')
    plt.title('Area Calculation Methods Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 9. Solidity distribution
    plt.subplot(3, 4, 9)
    if all_solidities:
        plt.hist(all_solidities, bins=30, alpha=0.7, color='brown', edgecolor='black')
        plt.xlabel('Solidity')
        plt.ylabel('Frequency')
        plt.title('Solidity Distribution')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No Solidity Data', ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('Solidity Distribution')
    
    # 10. Circularity distribution
    plt.subplot(3, 4, 10)
    if all_circularities:
        plt.hist(all_circularities, bins=30, alpha=0.7, color='teal', edgecolor='black')
        plt.xlabel('Circularity')
        plt.ylabel('Frequency')
        plt.title('Circularity Distribution')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No Circularity Data', ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('Circularity Distribution')
    
    # 11. Area vs Aspect Ratio scatter plot
    plt.subplot(3, 4, 11)
    plt.scatter(all_aspect_ratios, all_contour_areas, alpha=0.6, s=30, c='darkorange')
    plt.xlabel('Aspect Ratio')
    plt.ylabel('Area (pixels²)')
    plt.title('Area vs Aspect Ratio')
    plt.grid(True, alpha=0.3)
    
    # 12. Statistical summary text
    plt.subplot(3, 4, 12)
    plt.axis('off')
    
    stats_text = f"""Statistical Summary:
    
Total Connected Components: {len(all_contour_areas)}
False Positive Regions Filtered: {sum(stat['false_positive_regions_count'] for stat in image_stats)}

Area Statistics (pixels²):
  Mean: {np.mean(all_contour_areas):.2f}
  Median: {np.median(all_contour_areas):.2f}
  Std Dev: {np.std(all_contour_areas):.2f}
  Min: {np.min(all_contour_areas):.2f}
  Max: {np.max(all_contour_areas):.2f}

Perimeter Statistics (pixels):
  Mean: {np.mean(all_perimeters):.2f}
  Std Dev: {np.std(all_perimeters):.2f}

Aspect Ratio Statistics:
  Mean: {np.mean(all_aspect_ratios):.2f}
  Std Dev: {np.std(all_aspect_ratios):.2f}
"""
    
    if all_solidities:
        stats_text += f"""
Solidity Statistics:
  Mean: {np.mean(all_solidities):.3f}
  Std Dev: {np.std(all_solidities):.3f}
"""
    
    if all_circularities:
        stats_text += f"""
Circularity Statistics:
  Mean: {np.mean(all_circularities):.3f}
  Std Dev: {np.std(all_circularities):.3f}
"""
    
    plt.text(0.05, 0.95, stats_text, fontsize=9, verticalalignment='top', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8), transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'area_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return image_stats, all_contour_areas, all_perimeters, all_aspect_ratios

def save_data_to_files(all_data, image_stats, all_contour_areas, all_perimeters, all_aspect_ratios, output_dir):
    """Save data to files"""
    # Save detailed data with component features
    detailed_data = []
    for data in all_data:
        components_info = data.get('components_info', [])
        
        # 直接从components_info获取详细数据
        for i, comp in enumerate(components_info):
            area = comp['area']
            perimeter = cv2.arcLength(comp['contour'], True)
            aspect_ratio = comp['aspect_ratio']
            x, y, w, h = comp['bbox']
            bound_area = w * h
            min_rect = comp['min_rect']
            min_area = min_rect[1][0] * min_rect[1][1]
            
            detailed_data.append({
                'filename': data['filename'],
                'component_id': i + 1,
                'contour_area_pixel2': area,
                'perimeter_pixel': perimeter,
                'aspect_ratio': aspect_ratio,
                'bounding_rect_area_pixel2': bound_area,
                'min_rect_area_pixel2': min_area,
                'solidity': comp['solidity'],
                'circularity': comp['circularity'],
                'center_x': comp['center'][0],
                'center_y': comp['center'][1]
            })
        
        # 如果没有components_info，使用传统数据格式（向后兼容）
        if not components_info and len(data['contour_areas']) > 0:
            for i, (area, perim, aspect, bound_area, min_area) in enumerate(zip(
                data['contour_areas'], data['perimeters'], data['aspect_ratios'],
                data['bounding_rect_areas'], data['min_rect_areas'])):
                detailed_data.append({
                    'filename': data['filename'],
                    'component_id': i + 1,
                    'contour_area_pixel2': area,
                    'perimeter_pixel': perim,
                    'aspect_ratio': aspect,
                    'bounding_rect_area_pixel2': bound_area,
                    'min_rect_area_pixel2': min_area,
                    'solidity': data.get('solidities', [0])[i] if i < len(data.get('solidities', [])) else 0,
                    'circularity': data.get('circularities', [0])[i] if i < len(data.get('circularities', [])) else 0,
                    'center_x': 0,
                    'center_y': 0
                })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv(os.path.join(output_dir, 'detailed_component_data.csv'), index=False, encoding='utf-8-sig')
    
    # Save image-level statistics
    image_df = pd.DataFrame(image_stats)
    image_df.to_csv(os.path.join(output_dir, 'image_statistics.csv'), index=False, encoding='utf-8-sig')
    
    # Collect all shape feature data
    all_solidities = []
    all_circularities = []
    for data in all_data:
        all_solidities.extend(data.get('solidities', []))
        all_circularities.extend(data.get('circularities', []))
    
    # Save overall statistics
    summary_stats = {
        'total_components': len(all_contour_areas),
        'total_images': len(image_stats),
        'total_false_positive_regions': sum(stat.get('false_positive_regions_count', 0) for stat in image_stats),
        'contour_area_mean_pixel2': np.mean(all_contour_areas),
        'contour_area_median_pixel2': np.median(all_contour_areas),
        'contour_area_std_pixel2': np.std(all_contour_areas),
        'contour_area_min_pixel2': np.min(all_contour_areas) if all_contour_areas else 0,
        'contour_area_max_pixel2': np.max(all_contour_areas) if all_contour_areas else 0,
        'perimeter_mean_pixel': np.mean(all_perimeters),
        'perimeter_median_pixel': np.median(all_perimeters),
        'perimeter_std_pixel': np.std(all_perimeters),
        'aspect_ratio_mean': np.mean(all_aspect_ratios),
        'aspect_ratio_median': np.median(all_aspect_ratios),
        'aspect_ratio_std': np.std(all_aspect_ratios),
        'aspect_ratio_min': np.min(all_aspect_ratios) if all_aspect_ratios else 0,
        'aspect_ratio_max': np.max(all_aspect_ratios) if all_aspect_ratios else 0
    }
    
    # Add shape feature statistics if available
    if all_solidities:
        summary_stats.update({
            'solidity_mean': np.mean(all_solidities),
            'solidity_median': np.median(all_solidities),
            'solidity_std': np.std(all_solidities),
            'solidity_min': np.min(all_solidities),
            'solidity_max': np.max(all_solidities)
        })
    
    if all_circularities:
        summary_stats.update({
            'circularity_mean': np.mean(all_circularities),
            'circularity_median': np.median(all_circularities),
            'circularity_std': np.std(all_circularities),
            'circularity_min': np.min(all_circularities),
            'circularity_max': np.max(all_circularities)
        })
    
    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv(os.path.join(output_dir, 'summary_statistics.csv'), index=False, encoding='utf-8-sig')

def main():
    args = parse_args()
    
    # Check input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist!")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get image files
    image_files = get_image_files(args.input_dir, args.image_exts)
    if not image_files:
        print(f"No image files found in '{args.input_dir}' with extensions: {args.image_exts}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Load UNet model
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
    
    # Process images
    successful = 0
    failed = 0
    failed_files = []
    all_data = []
    
    print("Analyzing images...")
    print(f"智能误报过滤: {'启用' if args.enable_false_positive_filter else '禁用'}")
    print(f"智能连通域合并: {'启用' if args.enable_component_merge else '禁用'}")
    
    for image_path in tqdm(image_files, desc="Processing", unit="image"):
        success, result = process_single_image(image_path, net, device, args)
        
        if success:
            successful += 1
            all_data.append(result)
        else:
            failed += 1
            failed_files.append((image_path, result))
            print(f"\nFailed to process {image_path}: {result}")
    
    # Generate analysis results
    if all_data:
        print("Generating analysis plots...")
        image_stats, all_contour_areas, all_perimeters, all_aspect_ratios = create_analysis_plots(all_data, args.output_dir)
        
        print("Saving data files...")
        save_data_to_files(all_data, image_stats, all_contour_areas, all_perimeters, all_aspect_ratios, args.output_dir)
    
    # Output summary
    print(f"\n{'='*60}")
    print(f"Analysis completed!")
    print(f"{'='*60}")
    print(f"Total images: {len(image_files)}")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    
    if all_data:
        total_components = sum(data['component_count'] for data in all_data)
        total_false_positive_regions = sum(data.get('false_positive_regions_count', 0) for data in all_data)
        print(f"\nConnected Components Analysis:")
        print(f"  Total connected components: {total_components}")
        print(f"  Average components per image: {total_components/successful:.2f}")
        print(f"  False positive regions filtered: {total_false_positive_regions}")
        
        all_areas = []
        all_solidities = []
        all_circularities = []
        for data in all_data:
            all_areas.extend(data['contour_areas'])
            all_solidities.extend(data.get('solidities', []))
            all_circularities.extend(data.get('circularities', []))
        
        if all_areas:
            print(f"\nArea Statistics (pixels²):")
            print(f"  Mean: {np.mean(all_areas):.2f}")
            print(f"  Median: {np.median(all_areas):.2f}")
            print(f"  Min: {np.min(all_areas):.2f}")
            print(f"  Max: {np.max(all_areas):.2f}")
            print(f"  Std Dev: {np.std(all_areas):.2f}")
            
        # 收集所有长宽比数据
        all_aspect_ratios_collected = []
        for data in all_data:
            all_aspect_ratios_collected.extend(data['aspect_ratios'])
            
        if all_aspect_ratios_collected:
            print(f"\nAspect Ratio Statistics (长边/短边):")
            print(f"  Mean: {np.mean(all_aspect_ratios_collected):.3f}")
            print(f"  Median: {np.median(all_aspect_ratios_collected):.3f}")
            print(f"  Min: {np.min(all_aspect_ratios_collected):.3f}")
            print(f"  Max: {np.max(all_aspect_ratios_collected):.3f}")
            print(f"  Std Dev: {np.std(all_aspect_ratios_collected):.3f}")
            
        if all_solidities:
            print(f"\nSolidity Statistics:")
            print(f"  Mean: {np.mean(all_solidities):.3f}")
            print(f"  Median: {np.median(all_solidities):.3f}")
            print(f"  Min: {np.min(all_solidities):.3f}")
            print(f"  Max: {np.max(all_solidities):.3f}")
            
        if all_circularities:
            print(f"\nCircularity Statistics:")
            print(f"  Mean: {np.mean(all_circularities):.3f}")
            print(f"  Median: {np.median(all_circularities):.3f}")
            print(f"  Min: {np.min(all_circularities):.3f}")
            print(f"  Max: {np.max(all_circularities):.3f}")
    
    print(f"\nProcessing Parameters:")
    print(f"  连通域预过滤最小面积: {args.min_component_area}")
    print(f"  智能误报过滤: {'启用' if args.enable_false_positive_filter else '禁用'}")
    if args.enable_false_positive_filter:
        print(f"    误报密度阈值: {args.fp_density_threshold}")
        print(f"    误报面积阈值: {args.fp_area_threshold}")
        print(f"    误报评分阈值: {args.fp_score_threshold}")
    print(f"  智能连通域合并: {'启用' if args.enable_component_merge else '禁用'}")
    if args.enable_component_merge:
        print(f"    合并距离阈值: {args.merge_distance}")
        print(f"    合并角度阈值: {args.merge_angle_threshold}度")
    
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - area_analysis.png: Connected components analysis plots")
    print(f"  - detailed_component_data.csv: Detailed component features data")
    print(f"  - image_statistics.csv: Image-level statistics")
    print(f"  - summary_statistics.csv: Overall statistical summary")
    
    if failed_files:
        print(f"\nFailed files:")
        for file_path, error in failed_files:
            print(f"  - {file_path}: {error}")

if __name__ == '__main__':
    main() 