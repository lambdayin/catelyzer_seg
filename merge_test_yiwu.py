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
    parser.add_argument('--fp-remove-mode', choices=['extract', 'remove'], default='remove',
                       help="误报处理模式：extract=提取内部组件，remove=直接去除（推荐）")
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


def analyze_connected_components(mask):
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
    # actual_area = np.sum(component_mask > 0)
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
    # 通过凸包缺陷(convexity defects)来评估轮廓的复杂程度
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
    
    优化后的判断逻辑：
    1. 使用专门的误报面积阈值，而不是基于正常催化剂max_area
    2. 使用最小外接矩形密度，对倾斜催化剂更友好
    3. 去除内部空洞检测，避免误杀有空洞的正常催化剂
    4. 保留轮廓复杂度检测，识别真正不规则的误报区域
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


def extract_internal_components(false_positive_mask, args):
    """
    从误报的大区域中提取内部真正的催化剂组件
    
    核心算法：多尺度形态学分离
    """
    # 1. 使用开运算分离粘连的组件
    # 逐步增大核的尺寸，直到能够有效分离
    extracted_components = []
    
    for kernel_size in [3, 5, 7, 9]:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # 开运算：先腐蚀后膨胀，分离粘连区域
        opened_mask = cv2.morphologyEx(false_positive_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 找到分离后的连通域
        num_labels, labeled_mask = cv2.connectedComponents(opened_mask)
        
        for label in range(1, num_labels):
            component_mask = (labeled_mask == label).astype(np.uint8)
            area = cv2.countNonZero(component_mask)
            
            # 检查是否为合理尺寸的催化剂
            if args.min_area * 0.5 <= area <= args.max_area:
                # 计算基本特征
                contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 0:
                    contour = contours[0]
                    
                    # 基本几何特征计算
                    min_rect = cv2.minAreaRect(contour)
                    width, height = min_rect[1]
                    if width > 0 and height > 0:
                        aspect_ratio = max(width, height) / min(width, height)
                        
                        # 检查长宽比是否合理
                        if aspect_ratio <= args.max_aspect_ratio * 1.5:  # 稍微放宽标准
                            x, y, w, h = cv2.boundingRect(contour)
                            
                            # 计算实心度
                            hull = cv2.convexHull(contour)
                            hull_area = cv2.contourArea(hull)
                            solidity = area / hull_area if hull_area > 0 else 0
                            
                            # 计算圆形度
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
                                'label': f'extracted_{label}_{kernel_size}',
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
                            
                            extracted_components.append(component_info)
        
        # 如果当前尺寸已经能够有效分离，就不需要继续尝试更大的核
        if len(extracted_components) > 0:
            break
    
    return extracted_components


def intelligent_component_filtering(components_info, args):
    """
    智能连通域过滤：识别并处理UNet误报的大区域
    
    这是核心创新算法，能够：
    1. 识别UNet误报的大区域
    2. 根据模式选择：直接去除 或 提取内部组件
    3. 保留正常尺寸的连通域
    4. 返回误报区域信息（用于可视化）
    """
    filtered_components = []
    extracted_components = []
    false_positive_regions = []  # 新增：保存误报区域信息
    removed_count = 0
    
    mode_desc = "直接去除" if args.fp_remove_mode == 'remove' else "提取内部组件"
    print(f"\n开始智能连通域过滤，初始连通域数量: {len(components_info)}")
    print(f"误报处理模式: {mode_desc}")
    print(f"误报区域可视化: {'启用' if args.show_false_positive else '禁用'}")
    
    for comp in components_info:
        # 计算密度特征
        density_info = calculate_region_density(comp['mask'])
        
        # 判断是否为误报大区域
        if is_false_positive_region(comp, density_info, args):
            print(f"🚫 检测到误报大区域: 面积={comp['area']}, 最小外接矩形密度={density_info['bbox_density']:.3f}, 轮廓复杂度={density_info['complexity_score']}")
            
            # 保存误报区域信息（用于可视化）
            false_positive_regions.append({
                'mask': comp['mask'],
                'contour': comp['contour'],
                'area': comp['area'],
                'density': density_info['bbox_density'],
                'complexity': density_info['complexity_score']
            })
            
            if args.fp_remove_mode == 'remove':
                # 直接去除误报区域
                removed_count += 1
                print(f"  ❌ 直接去除该误报区域")
            else:
                # 从误报区域中提取真实组件
                internal_components = extract_internal_components(comp['mask'], args)
                extracted_components.extend(internal_components)
                print(f"  ✅ 从误报区域提取到 {len(internal_components)} 个内部组件")
            
        else:
            # 保留正常连通域
            filtered_components.append(comp)
    
    # 合并过滤后的连通域和提取的组件
    final_components = filtered_components + extracted_components
    
    if args.fp_remove_mode == 'remove':
        print(f"智能过滤完成: 保留正常组件 {len(filtered_components)} 个，去除误报区域 {removed_count} 个")
    else:
        print(f"智能过滤完成: 保留正常组件 {len(filtered_components)} 个，提取内部组件 {len(extracted_components)} 个")
    
    print(f"最终连通域数量: {len(final_components)}")
    
    return final_components, false_positive_regions


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
        if comp['aspect_ratio'] < args.min_aspect_ratio * 0.8:  # 更宽松
            anomaly_score += 2
            anomaly_reasons.append('aspect ratio is too small')
        elif comp['aspect_ratio'] > args.max_aspect_ratio * 1.2:  # 更宽松
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
        
        # 5. 综合评分判断
        comp['anomaly_score'] = anomaly_score
        comp['anomaly_reasons'] = anomaly_reasons
        
        # 使用评分制度进行分类
        if anomaly_score <= 1:  # 正常或轻微异常
            normal_components.append(comp)
        elif anomaly_score >= 3:  # 明显异常
            # 区分异物和异形催化剂
            if (comp['area'] < args.min_area * 1.5 or  # 面积较小
                comp['circularity'] > 0.7 or          # 较圆
                'shape is too circular' in anomaly_reasons):
                foreign_objects.append(comp)
            else:
                deformed_catalysts.append(comp)
        else:  # 中等异常 (score = 2)
            # 更保守的分类，倾向于归类为正常
            if (comp['circularity'] > 0.8 or  # 只有非常圆的才认为是异物
                comp['area'] < args.min_area * 0.5):  # 或者面积极小
                foreign_objects.append(comp)
            else:
                normal_components.append(comp)  # 归类为正常
    
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
    mask_clean = cv2.erode(mask_combined, kernel, iterations=2)
    mask_clean = cv2.dilate(mask_clean, kernel, iterations=2)
    mask_clean = cv2.dilate(mask_clean, kernel, iterations=1)
    mask_clean = cv2.erode(mask_clean, kernel, iterations=1)

    # 使用开运算去除小的噪声点
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    # 使用闭运算填充内部的小孔洞
    # mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    
    # 过滤掉面积过小的连通域（去除折叠催化剂的小露出部分等噪声）
    mask_filtered = filter_small_components(mask_clean, args.min_component_area)
    
    # 连通域分析
    components_info = analyze_connected_components(mask_filtered)
    
    # 识别并处理UNet误报的大区域，从中提取真正的催化剂
    false_positive_regions = []
    if args.enable_false_positive_filter:
        components_info, false_positive_regions = intelligent_component_filtering(components_info, args)
    
    # 智能连通域合并（可选）
    if args.enable_component_merge:
        components_info = merge_connected_components(
            components_info, args.merge_distance, args.merge_angle_threshold
        )
    
    # 异常分类
    classification_result = classify_anomalies(components_info, original_image.shape, args)
    
    return classification_result, mask_filtered, false_positive_regions


def visualize_results(original_image, classification_result, anomaly_mask, false_positive_regions=None, show_false_positive=False):
    """
    生成可视化结果
    显示整体催化剂连通域mask叠加效果，用不同颜色标注不同类型
    可选显示误报区域的半透明mask
    """
    vis_image = original_image.copy()
    
    # 创建彩色mask
    colored_mask = np.zeros_like(original_image)
    
    # 颜色定义 - 使用更鲜艳的颜色
    colors = {
        'foreign_objects': (0, 0, 255),      # 鲜红色 - 异物
        'deformed_catalysts': (0, 128, 255), # 鲜橙色 - 异形催化剂  
        'normal': (0, 255, 0)                # 鲜绿色 - 正常催化剂
    }
    
    # 标签文本
    labels = {
        'foreign_objects': 'foreign_objects',
        'deformed_catalysts': 'deformed_catalysts',
        'normal': 'normal'
    }
    
    # 绘制所有连通域的mask
    for category, components in classification_result.items():
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
            rect_points = np.intp(rect_points)
            cv2.drawContours(vis_image, [rect_points], -1, color, 2)
            
            # 添加中文标签
            center_x, center_y = comp['center']
            
            # 添加文字背景
            # text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            # cv2.rectangle(vis_image, (center_x-text_size[0]//2-5, center_y-text_size[1]-10), 
            #              (center_x+text_size[0]//2+5, center_y-5), (255, 255, 255), -1)
            # cv2.rectangle(vis_image, (center_x-text_size[0]//2-5, center_y-text_size[1]-10), 
            #              (center_x+text_size[0]//2+5, center_y-5), color, 2)
            
            # # 添加标签文字
            # cv2.putText(vis_image, label_text, (center_x-text_size[0]//2, center_y-8), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # 添加面积信息
            # area_text = f"area:{comp['area']}"
            # area_size = cv2.getTextSize(area_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            # cv2.rectangle(vis_image, (center_x-area_size[0]//2-3, center_y+5), 
            #              (center_x+area_size[0]//2+3, center_y+area_size[1]+8), (255, 255, 255), -1)
            # cv2.putText(vis_image, area_text, (center_x-area_size[0]//2, center_y+area_size[1]+5), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # 绘制误报区域（如果启用且有误报区域）
    if show_false_positive and false_positive_regions:
        # 创建误报区域的mask
        fp_mask = np.zeros_like(original_image)
        fp_color = (128, 0, 128)  # 紫色表示误报区域
        
        for fp_region in false_positive_regions:
            # 计算最小外接矩形
            min_rect = cv2.minAreaRect(fp_region['contour'])
            rect_points = cv2.boxPoints(min_rect)
            rect_points = np.int0(rect_points)
            
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
    
    # 添加统计信息背景
    # stats_bg_height = 100
    # stats_bg = np.ones((stats_bg_height, mask_overlay.shape[1], 3), dtype=np.uint8) * 240
    
    # 添加统计信息
    stats_text = [
        f"detection results:",
        f"foreign objects: {len(classification_result['foreign_objects'])}",
        f"deformed catalysts: {len(classification_result['deformed_catalysts'])}",
        f"normal catalysts: {len(classification_result['normal'])}"
    ]
    
    # for i, text in enumerate(stats_text):
    #     color = (0, 0, 0) if i == 0 else colors[list(colors.keys())[i-1]] if i <= 3 else (0, 0, 0)
    #     weight = 2 if i == 0 else 1
    #     cv2.putText(stats_bg, text, (15, 20 + i*20), 
    #                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, weight)
    
    # 将统计信息叠加到图像顶部
    # final_result = np.vstack([stats_bg, mask_overlay])
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
        mode_desc = "直接去除" if args.fp_remove_mode == 'remove' else "提取内部组件"
        print(f"  误报处理模式: {mode_desc}")
        print(f"  误报密度阈值: {args.fp_density_threshold}")
        print(f"  误报面积阈值: {args.fp_area_threshold}")
        print(f"  误报评分阈值: {args.fp_score_threshold}")
        print(f"  误报区域可视化: {'启用' if args.show_false_positive else '禁用'}")


if __name__ == '__main__':
    main() 