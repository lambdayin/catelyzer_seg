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

import torch
from utils.data_loading import SelfDataset
from unet import UNet

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  
rcParams['axes.unicode_minus'] = False

def parse_args():
    parser = argparse.ArgumentParser(description="分析连通域最小外接矩形面积分布")
    parser.add_argument('model', type=str, help="UNet模型检查点文件 (.pth)")
    parser.add_argument('--input-dir', default='./test_0527/imgdata', type=str, help="输入图像目录")
    parser.add_argument('--output-dir', default='./area_analysis_results', type=str, help="分析结果输出目录")
    parser.add_argument('--grid-len', default=36, type=float, help="每个网格的物理长度(mm)")
    parser.add_argument('--grid-pixel', default=651, type=float, help="每个网格的像素距离")
    parser.add_argument('--image-exts', default=['jpg', 'jpeg', 'png', 'bmp', 'tiff'], nargs='+', 
                       help="要处理的图像文件扩展名")
    return parser.parse_args()

def get_image_files(input_dir, extensions):
    """获取所有图像文件"""
    image_files = []
    for ext in extensions:
        pattern = os.path.join(input_dir, f"*.{ext}")
        image_files.extend(glob.glob(pattern, recursive=False))
        pattern = os.path.join(input_dir, f"*.{ext.upper()}")
        image_files.extend(glob.glob(pattern, recursive=False))
    return sorted(list(set(image_files)))

def pre_process_single_image(image_path):
    """预处理单张图像，与merge_test.py保持一致"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    mask_filter = yuv[:, :, 0] < 15
    mask_filter = mask_filter.astype(np.uint8)
    mask_eroded = cv2.dilate(mask_filter, np.ones((5, 5), np.uint8), iterations=2)
    mask_eroded = cv2.erode(mask_eroded, np.ones((5, 5), np.uint8), iterations=2)
        
    image[mask_eroded == 1] = [255, 255, 255]
    return image, mask_eroded

def inference_unet_batch(net, device, image_path):
    """UNet推理，与merge_test.py保持一致"""
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

def analyze_connected_components(image_path, mm_per_pixel, mask_eroded, mask_unet):
    """分析连通域并收集面积数据"""
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    mask_eroded = 1 - mask_eroded
    mask_unet = cv2.resize(mask_unet, (orig_img.shape[1], orig_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask_check = mask_unet & mask_eroded
    mask_check = mask_check.astype(np.uint8)
    mask_check = cv2.erode(mask_check, np.ones((3, 3), np.uint8), iterations=2)
    mask_check = cv2.dilate(mask_check, np.ones((3, 3), np.uint8), iterations=2)
    mask_check = cv2.dilate(mask_check, np.ones((3, 3), np.uint8), iterations=1)
    mask_check = cv2.erode(mask_check, np.ones((3, 3), np.uint8), iterations=1)

    # 使用开运算去除小的噪声点
    mask_check = cv2.morphologyEx(mask_check, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

    _, labeled_mask = cv2.connectedComponents(mask_check)
    labeled_mask = np.uint8(labeled_mask)

    contours, __ = cv2.findContours(labeled_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    areas_pixel = []  # 像素面积
    areas_mm2 = []    # 实际面积(mm²)
    lengths_pixel = [] # 最大边长像素
    lengths_mm = []    # 最大边长mm
    
    for contour in contours:
        if len(contour) < 5:  # 连通域太小，跳过
            continue
        min_rect = cv2.minAreaRect(contour)
        width, height = min_rect[1]
        
        # 计算面积
        area_pixel = width * height
        area_mm2 = area_pixel * (mm_per_pixel ** 2)
        
        # 计算最大边长
        max_length_pixel = max(width, height)
        max_length_mm = max_length_pixel * mm_per_pixel
        
        areas_pixel.append(area_pixel)
        areas_mm2.append(area_mm2)
        lengths_pixel.append(max_length_pixel)
        lengths_mm.append(max_length_mm)
    
    return areas_pixel, areas_mm2, lengths_pixel, lengths_mm

def process_single_image(image_path, net, device, mm_per_pixel):
    """处理单张图像并返回面积数据"""
    try:
        _, mask_eroded = pre_process_single_image(image_path)
        mask_unet = inference_unet_batch(net, device, image_path)
        areas_pixel, areas_mm2, lengths_pixel, lengths_mm = analyze_connected_components(
            image_path, mm_per_pixel, mask_eroded, mask_unet)
        
        filename = os.path.basename(image_path)
        return True, {
            'filename': filename,
            'areas_pixel': areas_pixel,
            'areas_mm2': areas_mm2,
            'lengths_pixel': lengths_pixel,
            'lengths_mm': lengths_mm,
            'component_count': len(areas_pixel)
        }
    except Exception as e:
        return False, str(e)

def create_analysis_plots(all_data, output_dir):
    """创建分析图表"""
    # 收集所有数据
    all_areas_mm2 = []
    all_lengths_mm = []
    image_stats = []
    
    for data in all_data:
        all_areas_mm2.extend(data['areas_mm2'])
        all_lengths_mm.extend(data['lengths_mm'])
        image_stats.append({
            'filename': data['filename'],
            'component_count': data['component_count'],
            'total_area': sum(data['areas_mm2']),
            'avg_area': np.mean(data['areas_mm2']) if data['areas_mm2'] else 0,
            'max_area': max(data['areas_mm2']) if data['areas_mm2'] else 0,
            'avg_length': np.mean(data['lengths_mm']) if data['lengths_mm'] else 0,
            'max_length': max(data['lengths_mm']) if data['lengths_mm'] else 0
        })
    
    # 创建图表
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 面积分布直方图
    plt.subplot(3, 3, 1)
    plt.hist(all_areas_mm2, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('连通域面积 (mm²)')
    plt.ylabel('频次')
    plt.title('连通域面积分布直方图')
    plt.grid(True, alpha=0.3)
    
    # 2. 面积分布箱线图
    plt.subplot(3, 3, 2)
    plt.boxplot(all_areas_mm2)
    plt.ylabel('连通域面积 (mm²)')
    plt.title('连通域面积分布箱线图')
    plt.grid(True, alpha=0.3)
    
    # 3. 长度分布直方图
    plt.subplot(3, 3, 3)
    plt.hist(all_lengths_mm, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.xlabel('最大边长 (mm)')
    plt.ylabel('频次')
    plt.title('连通域最大边长分布直方图')
    plt.grid(True, alpha=0.3)
    
    # 4. 每张图像的连通域数量
    plt.subplot(3, 3, 4)
    component_counts = [stat['component_count'] for stat in image_stats]
    plt.bar(range(len(component_counts)), component_counts, alpha=0.7, color='orange')
    plt.xlabel('图像索引')
    plt.ylabel('连通域数量')
    plt.title('每张图像的连通域数量')
    plt.grid(True, alpha=0.3)
    
    # 5. 面积vs长度散点图
    plt.subplot(3, 3, 5)
    plt.scatter(all_lengths_mm, all_areas_mm2, alpha=0.6, s=30)
    plt.xlabel('最大边长 (mm)')
    plt.ylabel('面积 (mm²)')
    plt.title('面积 vs 最大边长关系')
    plt.grid(True, alpha=0.3)
    
    # 6. 累积分布图
    plt.subplot(3, 3, 6)
    sorted_areas = np.sort(all_areas_mm2)
    cumulative = np.arange(1, len(sorted_areas) + 1) / len(sorted_areas)
    plt.plot(sorted_areas, cumulative, linewidth=2)
    plt.xlabel('连通域面积 (mm²)')
    plt.ylabel('累积概率')
    plt.title('面积累积分布函数')
    plt.grid(True, alpha=0.3)
    
    # 7. 对数尺度面积分布
    plt.subplot(3, 3, 7)
    log_areas = np.log10(np.array(all_areas_mm2) + 1e-6)
    plt.hist(log_areas, bins=30, alpha=0.7, color='purple', edgecolor='black')
    plt.xlabel('log₁₀(面积) (mm²)')
    plt.ylabel('频次')
    plt.title('面积分布（对数尺度）')
    plt.grid(True, alpha=0.3)
    
    # 8. 每张图像的总面积
    plt.subplot(3, 3, 8)
    total_areas = [stat['total_area'] for stat in image_stats]
    plt.bar(range(len(total_areas)), total_areas, alpha=0.7, color='red')
    plt.xlabel('图像索引')
    plt.ylabel('总面积 (mm²)')
    plt.title('每张图像的总连通域面积')
    plt.grid(True, alpha=0.3)
    
    # 9. 统计摘要文本
    plt.subplot(3, 3, 9)
    plt.axis('off')
    stats_text = f"""统计摘要：
    
总连通域数量: {len(all_areas_mm2)}
面积统计 (mm²):
  平均值: {np.mean(all_areas_mm2):.2f}
  中位数: {np.median(all_areas_mm2):.2f}
  标准差: {np.std(all_areas_mm2):.2f}
  最小值: {np.min(all_areas_mm2):.2f}
  最大值: {np.max(all_areas_mm2):.2f}

长度统计 (mm):
  平均值: {np.mean(all_lengths_mm):.2f}
  中位数: {np.median(all_lengths_mm):.2f}
  标准差: {np.std(all_lengths_mm):.2f}
  最小值: {np.min(all_lengths_mm):.2f}
  最大值: {np.max(all_lengths_mm):.2f}
"""
    plt.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'area_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return image_stats, all_areas_mm2, all_lengths_mm

def save_data_to_files(all_data, image_stats, all_areas_mm2, all_lengths_mm, output_dir):
    """保存数据到文件"""
    # 保存详细数据
    detailed_data = []
    for data in all_data:
        for i, (area_mm2, length_mm) in enumerate(zip(data['areas_mm2'], data['lengths_mm'])):
            detailed_data.append({
                'filename': data['filename'],
                'component_id': i + 1,
                'area_mm2': area_mm2,
                'max_length_mm': length_mm
            })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv(os.path.join(output_dir, 'detailed_component_data.csv'), index=False, encoding='utf-8-sig')
    
    # 保存图像级统计
    image_df = pd.DataFrame(image_stats)
    image_df.to_csv(os.path.join(output_dir, 'image_statistics.csv'), index=False, encoding='utf-8-sig')
    
    # 保存总体统计
    summary_stats = {
        'total_components': len(all_areas_mm2),
        'total_images': len(image_stats),
        'area_mean_mm2': np.mean(all_areas_mm2),
        'area_median_mm2': np.median(all_areas_mm2),
        'area_std_mm2': np.std(all_areas_mm2),
        'area_min_mm2': np.min(all_areas_mm2) if all_areas_mm2 else 0,
        'area_max_mm2': np.max(all_areas_mm2) if all_areas_mm2 else 0,
        'length_mean_mm': np.mean(all_lengths_mm),
        'length_median_mm': np.median(all_lengths_mm),
        'length_std_mm': np.std(all_lengths_mm),
        'length_min_mm': np.min(all_lengths_mm) if all_lengths_mm else 0,
        'length_max_mm': np.max(all_lengths_mm) if all_lengths_mm else 0
    }
    
    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv(os.path.join(output_dir, 'summary_statistics.csv'), index=False, encoding='utf-8-sig')

def main():
    args = parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录 '{args.input_dir}' 不存在!")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取图像文件
    image_files = get_image_files(args.input_dir, args.image_exts)
    if not image_files:
        print(f"在 '{args.input_dir}' 中没有找到图像文件，支持的扩展名: {args.image_exts}")
        return
    
    print(f"找到 {len(image_files)} 张图像待处理")
    
    # 计算毫米每像素
    mm_per_pixel = float(args.grid_len / args.grid_pixel)
    print(f"像素到毫米转换比例: {mm_per_pixel:.6f} mm/pixel")
    
    # 加载UNet模型
    print("加载UNet模型...")
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
        print(f"加载模型时出错: {e}")
        return
    
    # 处理图像
    successful = 0
    failed = 0
    failed_files = []
    all_data = []
    
    print("分析图像中...")
    for image_path in tqdm(image_files, desc="处理中", unit="图像"):
        success, result = process_single_image(image_path, net, device, mm_per_pixel)
        
        if success:
            successful += 1
            all_data.append(result)
        else:
            failed += 1
            failed_files.append((image_path, result))
            print(f"\n处理失败 {image_path}: {result}")
    
    # 生成分析结果
    if all_data:
        print("生成分析图表...")
        image_stats, all_areas_mm2, all_lengths_mm = create_analysis_plots(all_data, args.output_dir)
        
        print("保存数据文件...")
        save_data_to_files(all_data, image_stats, all_areas_mm2, all_lengths_mm, args.output_dir)
    
    # 输出总结
    print(f"\n{'='*60}")
    print(f"分析完成!")
    print(f"总图像数: {len(image_files)}")
    print(f"成功处理: {successful}")
    print(f"处理失败: {failed}")
    
    if all_data:
        total_components = sum(data['component_count'] for data in all_data)
        print(f"总连通域数: {total_components}")
        print(f"平均每张图像连通域数: {total_components/successful:.2f}")
        
        all_areas = []
        for data in all_data:
            all_areas.extend(data['areas_mm2'])
        
        if all_areas:
            print(f"连通域面积统计 (mm²):")
            print(f"  平均值: {np.mean(all_areas):.2f}")
            print(f"  中位数: {np.median(all_areas):.2f}")
            print(f"  最小值: {np.min(all_areas):.2f}")
            print(f"  最大值: {np.max(all_areas):.2f}")
    
    print(f"结果保存到: {args.output_dir}")
    print(f"  - area_analysis.png: 分析图表")
    print(f"  - detailed_component_data.csv: 详细连通域数据")
    print(f"  - image_statistics.csv: 图像级统计")
    print(f"  - summary_statistics.csv: 总体统计摘要")
    
    if failed_files:
        print(f"\n失败的文件:")
        for file_path, error in failed_files:
            print(f"  - {file_path}: {error}")

if __name__ == '__main__':
    main() 