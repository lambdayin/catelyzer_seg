# 催化剂异物异形检测系统

一个基于深度学习UNet模型的工业催化剂图像分析系统，专门用于检测催化剂中的异物和异形缺陷。

## 📋 项目概述

本项目是一个完整的工业视觉检测解决方案，主要用于催化剂生产质量控制。系统通过UNet深度学习模型进行图像分割，结合先进的图像处理算法和统计分析方法，实现对催化剂异物和异形的智能检测与分类。

### 🎯 主要功能

- **图像分割**: 基于UNet模型的精准像素级分割
- **异物检测**: 智能识别催化剂中的外来物质
- **异形检测**: 检测变形、破损的催化剂颗粒
- **连通域分析**: 深度分析分割区域的几何特征
- **智能误报过滤**: 采用多重算法减少误检
- **统计分析**: 全面的数据统计和可视化
- **批量处理**: 支持大批量图像自动化处理

## 🚀 核心特性

### 1. 智能检测算法
- **UNet深度学习模型**: 高精度像素级分割
- **连通域合并**: 智能合并被分割的单一催化剂
- **多特征综合判断**: 面积、长宽比、实心度等多维度分析
- **自适应阈值**: 根据统计分布动态调整检测参数

### 2. 误报抑制技术  
- **密度检测**: 基于像素密度的误报识别
- **形态学分析**: 通过形状特征过滤噪声
- **评分机制**: 多因子综合评分系统
- **边缘检测**: 排除图像边缘的干扰区域

### 3. 可视化与分析
- **结果可视化**: 直观的检测结果标注和展示
- **统计报告**: 详细的数据分析和图表生成
- **批量分析**: 支持大规模数据集的统计分析
- **导出功能**: 多格式结果数据导出

## 🛠️ 技术栈

- **深度学习框架**: PyTorch
- **计算机视觉**: OpenCV
- **数据处理**: NumPy, SciPy
- **可视化**: Matplotlib, Seaborn
- **实验管理**: Weights & Biases (wandb)
- **数据格式**: LMDB, PIL/Pillow
- **开发语言**: Python 3.x

## 📦 安装配置

### 环境要求

- Python >= 3.7
- CUDA >= 10.0 (GPU加速，可选)
- 内存 >= 8GB
- 显存 >= 4GB (使用GPU时)

### 安装步骤

1. **克隆项目**
```bash
git clone <project-url>
cd catelyzer_seg
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **准备预训练模型**
```bash
# 将预训练模型放置到 pretrained/ 目录
mkdir -p pretrained
# 下载或复制模型文件到 pretrained/twoimages_epoch1000.pth
```

4. **Docker部署 (可选)**
```bash
docker build -t catalyst-detector .
docker run --gpus all -v $(pwd):/workspace/unet catalyst-detector
```

## 🎮 使用方法

### 1. 异物异形检测

使用主检测脚本进行催化剂异物异形检测：

```bash
# 基础检测
python merge_test_yiwu.py pretrained/twoimages_epoch1000.pth \
  --input-dir ./data/test_images \
  --output-dir ./results

# 使用预配置脚本
bash 02_merge_test_yiwu.sh
```

#### 主要参数说明

**检测参数**:
- `--min-component-area`: 连通域最小面积阈值 (默认: 500)
- `--min-area`: 异物最小面积 (默认: 500)  
- `--max-area`: 异物最大面积 (默认: 20000)
- `--min-aspect-ratio`: 最小长宽比 (默认: 2.0)
- `--max-aspect-ratio`: 最大长宽比 (默认: 20.0)
- `--min-solidity`: 最小实心度 (默认: 0.8)

**智能合并参数**:
- `--enable-component-merge`: 启用连通域智能合并
- `--merge-distance`: 合并距离阈值 (默认: 20)
- `--merge-angle-threshold`: 合并角度阈值 (默认: 30度)

**误报过滤参数**:
- `--enable-false-positive-filter`: 启用智能误报过滤
- `--fp-density-threshold`: 密度阈值 (默认: 0.4)
- `--fp-area-threshold`: 面积阈值 (默认: 150000)
- `--fp-score-threshold`: 评分阈值 (默认: 3)

### 2. 区域分析

进行详细的连通域统计分析：

```bash
python area_analysis.py pretrained/twoimages_epoch1000.pth \
  --input-dir ./data/test_images \
  --output-dir ./analysis_results \
  --enable-false-positive-filter \
  --enable-component-merge
```

### 3. 模型训练

训练新的UNet模型：

```bash
# 基础训练
python train.py --epochs 100 --batch-size 2 --learning-rate 1e-4

# 催化剂专用训练
python train_catelyzer.py --data-dir ./data/training --epochs 200
```

### 4. 单张图像预测

对单张图像进行预测：

```bash
python predict.py --model pretrained/twoimages_epoch1000.pth \
  --input image.jpg --output result.png --viz
```

## 📊 输出结果

### 1. 检测结果文件

- **可视化图像**: 标注了检测结果的原图
- **统计数据**: CSV格式的详细检测数据
- **分析报告**: 包含统计图表的综合报告

### 2. 文件结构示例

```
results/
├── vis_images/           # 可视化结果图像
│   ├── image1_result.jpg
│   └── image2_result.jpg
├── statistics/           # 统计数据
│   ├── detection_summary.csv
│   └── component_details.csv
└── analysis/            # 分析报告
    ├── distribution_plots.png
    └── statistical_report.html
```

### 3. 检测类别

- **正常催化剂**: 符合规格的正常颗粒
- **异物**: 非催化剂的外来物质
- **异形**: 变形、破损的催化剂颗粒

## 🔧 配置文件

### 主要配置类 (DetectionConfig)

```python
@dataclass
class DetectionConfig:
    # 图像处理参数
    YUV_BRIGHTNESS_THRESHOLD: int = 15
    UNET_SCALE_FACTOR: float = 0.5
    
    # 连通域过滤参数  
    MIN_CONTOUR_POINTS: int = 10
    MIN_COMPONENT_FOR_ORIENTATION: int = 5
    
    # 形态学参数
    EROSION_KERNEL_SIZE: Tuple[int, int] = (3, 3)
    DILATION_KERNEL_SIZE: Tuple[int, int] = (5, 5)
    
    # 几何特征阈值
    MIN_CONTOUR_LENGTH: int = 3
    CURVATURE_SCALE_FACTOR: float = 50
```

## 📁 项目结构

```
catelyzer_seg/
├── merge_test_yiwu.py      # 主检测系统
├── area_analysis.py        # 区域分析工具
├── train.py               # UNet训练脚本
├── predict.py             # 预测脚本
├── evaluate.py            # 模型评估
├── requirements.txt       # 依赖列表
├── Dockerfile            # Docker配置
├── unet/                 # UNet模型定义
│   ├── unet_model.py
│   └── unet_parts.py
├── utils/                # 工具函数
│   ├── data_loading.py   # 数据加载
│   ├── dice_score.py     # 损失函数
│   └── utils.py          # 通用工具
├── datasets/             # 数据集处理
├── scripts/              # 辅助脚本
│   ├── 01_merge_test.sh  # 长度测量脚本
│   └── 02_merge_test_yiwu.sh  # 异物检测脚本
├── data/                 # 数据目录
└── pretrained/           # 预训练模型
```

## 🔬 算法原理

### 1. UNet分割网络
- **编码器-解码器结构**: 有效提取多尺度特征
- **跳跃连接**: 保留细节信息
- **像素级预测**: 精确的边界分割

### 2. 连通域分析
- **轮廓提取**: 基于OpenCV的轮廓检测
- **特征计算**: 面积、周长、长宽比、实心度等
- **形态学处理**: 开运算、闭运算优化分割结果

### 3. 智能合并算法
- **距离判断**: 基于质心距离的邻近性分析
- **角度约束**: 主轴方向的一致性检验
- **面积比例**: 避免错误合并不同大小的区域

### 4. 误报过滤机制
- **密度特征**: 基于最小外接矩形的像素密度
- **形状复杂度**: 轮廓复杂性分析
- **综合评分**: 多因子加权评分系统

## ⚡ 性能优化

### 1. GPU加速
- 支持CUDA GPU加速推理
- 自动检测可用GPU设备
- 内存优化的批处理

### 2. 数据处理优化
- LMDB数据库支持大规模数据
- 多进程数据加载
- 内存映射减少IO开销

### 3. 算法优化
- 向量化数学运算
- 高效的图像处理算法
- 智能缓存机制

## 🐛 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 降低批处理大小或使用CPU
   export CUDA_VISIBLE_DEVICES=""
   ```

2. **模型加载失败**
   ```bash
   # 检查模型文件路径和完整性
   ls -la pretrained/twoimages_epoch1000.pth
   ```

3. **依赖包冲突**
   ```bash
   # 使用虚拟环境
   python -m venv catalyst_env
   source catalyst_env/bin/activate
   pip install -r requirements.txt
   ```

### 日志调试

```bash
# 启用详细日志
python merge_test_yiwu.py --log-level DEBUG
```

## 📈 更新日志

### v2.0 (最新)
- ✅ 新增智能连通域合并功能
- ✅ 优化误报过滤算法
- ✅ 增强可视化效果
- ✅ 支持批量统计分析

### v1.0
- ✅ 基础UNet分割功能
- ✅ 连通域特征提取
- ✅ 简单异常检测

## 🤝 贡献指南

1. Fork本项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本项目使用GNU General Public License v3.0许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📧 联系方式

如有问题或建议，请通过以下方式联系：

- 项目Issues: [GitHub Issues](项目GitHub地址/issues)
- 邮箱: your-email@example.com

## 🙏 致谢

- PyTorch团队提供的深度学习框架
- OpenCV社区的计算机视觉支持
- 所有贡献者和测试用户的宝贵反馈

---

*本项目致力于工业质量检测的智能化升级，为催化剂生产提供可靠的视觉检测解决方案。* 