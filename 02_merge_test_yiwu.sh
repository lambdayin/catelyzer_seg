#!/bin/bash

# =================================================================
# 催化剂异物异形检测脚本启动器 (merge_test_yiwu.py)
# 一键启动催化剂异物异形检测功能
# 
# 新增功能:
# - 智能连通域合并：解决UNet分割将单一催化剂分成多个小块的问题
# - 优化异常判断：采用评分制度替代硬阈值，大幅降低误报率
# - 小连通域过滤：去除折叠催化剂的小露出部分等噪声
# =================================================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# =================================================================
# 配置参数 (请根据实际情况修改)
# =================================================================

# 模型文件路径 (必需参数)
MODEL_PATH="./pretrained/twoimages_epoch1000.pth"

# 输入输出目录
INPUT_DIR="./data/catalyst_merge/origin_data"
OUTPUT_DIR="./data/catalyst_merge/vis_result_yiwu_v0003_0708_advanced_new_add_wq"

# 异物检测参数
MIN_COMPONENT_AREA=500 # 连通域预过滤最小面积阈值
MIN_AREA=500           # 最小连通域面积阈值
MAX_AREA=20000         # 最大连通域面积阈值
MIN_ASPECT_RATIO=1.5   # 最小长宽比阈值
MAX_ASPECT_RATIO=20.0  # 最大长宽比阈值
MIN_SOLIDITY=0.8       # 最小实心度阈值
EDGE_THRESHOLD=50      # 边缘区域阈值(像素)=>预设，没有用到

# 智能连通域合并参数
ENABLE_COMPONENT_MERGE=true  # 启用智能连通域合并 (true/false)
MERGE_DISTANCE=20            # 连通域合并距离阈值
MERGE_ANGLE_THRESHOLD=30     # 连通域合并角度阈值(度)

# 智能误报过滤参数
ENABLE_FP_FILTER=true       # 启用智能误报过滤算法 (true/false)
FP_DENSITY_THRESHOLD=0.4    # 误报判断密度阈值 (0.1-0.8, 越小越严格)
FP_AREA_THRESHOLD=5000      # 误报判断面积阈值 (绝对像素值, 针对误报大区域)
FP_SCORE_THRESHOLD=4       # 误报判断综合评分阈值 (1-5, 越小越严格)

SHOW_FALSE_POSITIVE=false   # 显示误报区域: true=在结果图中显示误报区域mask, false=不显示

# 🌟 弯曲度分析参数
ENABLE_CURVATURE_ANALYSIS=true  # 启用弯曲度分析：区分弯曲催化剂和直条状催化剂
CURVATURE_SCORE_THRESHOLD=35    # 弯曲度判断评分阈值（越小越严格，推荐范围：25-50）
SHOW_CURVATURE_DETAILS=true     # 显示弯曲度详细信息

# 支持的图像格式
IMAGE_EXTENSIONS="jpg jpeg png bmp tiff"

# GPU设备 (设置为空字符串使用CPU)
CUDA_DEVICE="0"

# =================================================================
# 预检查
# =================================================================

print_info "开始催化剂异物异形检测任务..."
print_info "=================================="

# 检查Python环境
if ! command -v python &> /dev/null; then
    print_error "Python未找到，请确保Python已安装并在PATH中"
    exit 1
fi

# 检查脚本文件
if [ ! -f "merge_test_yiwu.py" ]; then
    print_error "merge_test_yiwu.py文件未找到"
    exit 1
fi

# 检查模型文件
if [ ! -f "$MODEL_PATH" ]; then
    print_error "模型文件未找到: $MODEL_PATH"
    print_info "请检查模型文件路径是否正确"
    exit 1
fi

# 检查输入目录
if [ ! -d "$INPUT_DIR" ]; then
    print_error "输入目录不存在: $INPUT_DIR"
    exit 1
fi

# 统计输入图像数量
image_count=0
for ext in $IMAGE_EXTENSIONS; do
    count=$(find "$INPUT_DIR" -maxdepth 1 -name "*.$ext" -o -name "*.$(echo $ext | tr '[:lower:]' '[:upper:]')" | wc -l)
    image_count=$((image_count + count))
done

if [ $image_count -eq 0 ]; then
    print_warning "在输入目录中未找到支持的图像文件"
    print_info "支持的格式: $IMAGE_EXTENSIONS"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# =================================================================
# 打印配置信息
# =================================================================

print_success "预检查通过！"
echo ""
print_info "配置信息:"
echo "  模型文件: $MODEL_PATH"
echo "  输入目录: $INPUT_DIR"
echo "  输出目录: $OUTPUT_DIR"
echo "  图像数量: $image_count 张"
echo "  GPU设备: ${CUDA_DEVICE:-CPU模式}"
echo ""
print_info "检测参数:"
echo "  连通域预过滤最小面积: $MIN_COMPONENT_AREA"
echo "  最小面积阈值: $MIN_AREA"
echo "  最大面积阈值: $MAX_AREA"
echo "  长宽比范围: $MIN_ASPECT_RATIO - $MAX_ASPECT_RATIO"
echo "  最小实心度: $MIN_SOLIDITY"
echo "  边缘阈值: ${EDGE_THRESHOLD}px"
echo ""
print_info "智能合并参数:"
echo "  启用智能合并: $ENABLE_COMPONENT_MERGE"
if [ "$ENABLE_COMPONENT_MERGE" = "true" ]; then
    echo "  合并距离阈值: $MERGE_DISTANCE"
    echo "  合并角度阈值: ${MERGE_ANGLE_THRESHOLD}度"
fi
echo ""
print_info "🚀 智能误报过滤参数 (新功能):"
echo "  启用误报过滤: $ENABLE_FP_FILTER"
if [ "$ENABLE_FP_FILTER" = "true" ]; then
    echo "  处理模式: 直接去除 \(固定模式\)"
    echo "  密度阈值: $FP_DENSITY_THRESHOLD \(越小越严格\)"
    echo "  面积阈值: $FP_AREA_THRESHOLD \(像素数，超过此值视为误报\)"
    echo "  综合评分阈值: $FP_SCORE_THRESHOLD \(越小越严格\)"
    echo "  显示误报区域: $SHOW_FALSE_POSITIVE \(true=在结果图中显示, false=不显示\)"
fi
echo ""

# 询问是否继续
read -p "按回车键开始检测，或输入 'q' 退出: " confirm
if [ "$confirm" = "q" ] || [ "$confirm" = "Q" ]; then
    print_info "用户取消操作"
    exit 0
fi

# =================================================================
# 设置环境变量
# =================================================================

if [ -n "$CUDA_DEVICE" ]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
    print_info "设置GPU设备: $CUDA_DEVICE"
fi

# =================================================================
# 执行主程序
# =================================================================

print_info "开始异物异形检测..."
echo "=================================="

# 构建Python命令
python_cmd="python merge_test_yiwu.py \"$MODEL_PATH\" \
    --input-dir \"$INPUT_DIR\" \
    --output-dir \"$OUTPUT_DIR\" \
    --min-component-area $MIN_COMPONENT_AREA \
    --min-area $MIN_AREA \
    --max-area $MAX_AREA \
    --min-aspect-ratio $MIN_ASPECT_RATIO \
    --max-aspect-ratio $MAX_ASPECT_RATIO \
    --min-solidity $MIN_SOLIDITY \
    --edge-threshold $EDGE_THRESHOLD \
    --image-exts $IMAGE_EXTENSIONS"

# 添加智能合并参数
if [ "$ENABLE_COMPONENT_MERGE" = "true" ]; then
    python_cmd="$python_cmd \
        --enable-component-merge \
        --merge-distance $MERGE_DISTANCE \
        --merge-angle-threshold $MERGE_ANGLE_THRESHOLD"
fi

# 添加智能误报过滤参数
if [ "$ENABLE_FP_FILTER" = "true" ]; then
    python_cmd="$python_cmd \
        --enable-false-positive-filter \
        --fp-density-threshold $FP_DENSITY_THRESHOLD \
        --fp-area-threshold $FP_AREA_THRESHOLD \
        --fp-score-threshold $FP_SCORE_THRESHOLD"
    
    # 添加误报区域显示参数
    if [ "$SHOW_FALSE_POSITIVE" = "true" ]; then
        python_cmd="$python_cmd --show-false-positive"
    fi
fi

# 添加弯曲度分析参数
if [ "$ENABLE_CURVATURE_ANALYSIS" = "true" ]; then
    python_cmd="$python_cmd \
        --enable-curvature-analysis \
        --curvature-score-threshold $CURVATURE_SCORE_THRESHOLD"
    
    # 添加弯曲度详细信息显示参数
    if [ "$SHOW_CURVATURE_DETAILS" = "true" ]; then
        python_cmd="$python_cmd --show-curvature-details"
    fi
fi

# 记录开始时间
start_time=$(date +%s)

# 执行命令
if eval $python_cmd; then
    # 计算耗时
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    print_success "检测完成！"
    echo "=================================="
    print_info "总耗时: ${duration}秒"
    print_info "结果保存在: $OUTPUT_DIR"
    
    # 统计输出文件
    result_count=$(find "$OUTPUT_DIR" -name "*_yiwu_result.*" | wc -l)
    print_info "生成结果图像: $result_count 张"
    
    # 显示检测结果统计
    print_info "检测结果预览:"
    echo "  🔴 异物检测结果 - 红色标注"
    echo "  🟠 异形催化剂检测结果 - 橙色标注"
    echo "  🟢 正常催化剂检测结果 - 绿色标注"
    echo ""
    
else
    print_error "检测失败！"
    print_info "请检查："
    echo "  - 模型文件是否正确"
    echo "  - 输入图像是否有效"
    echo "  - GPU/CPU资源是否充足"
    echo "  - Python环境依赖是否完整"
    exit 1
fi

print_success "异物异形检测脚本执行完成！"

# =================================================================
# 结果说明
# =================================================================

echo ""
print_info "结果说明:"
echo "=================================="
echo "📋 输出文件说明："
echo "  - 文件名格式: {原文件名}_yiwu_result.png"
echo "  - 图像顶部显示检测统计信息"
echo "  - 连通域区域用彩色mask高亮显示"
echo ""
echo "🎨 颜色标注含义："
echo "  - 🔴 红色区域: 检测到的异物"
echo "  - 🟠 橙色区域: 检测到的异形催化剂"
echo "  - 🟢 绿色区域: 正常催化剂"
echo "  - 🟣 紫色区域: 误报区域 \(仅在启用SHOW_FALSE_POSITIVE时显示\)"
echo ""
echo "📊 可调节参数："
echo "  - 如需调整检测灵敏度，请修改脚本中的检测参数"
echo "  - 启用/禁用智能合并：修改 ENABLE_COMPONENT_MERGE 参数"
echo "  - 合并距离调节：修改 MERGE_DISTANCE \(推荐范围: 15-40\)"
echo "  - 合并角度调节：修改 MERGE_ANGLE_THRESHOLD \(推荐范围: 20-45度\)"
echo ""
echo "🚀 新功能 - 智能误报过滤（已优化）："
echo "  - 启用/禁用误报过滤：修改 ENABLE_FP_FILTER 参数"
echo "  - 处理模式：固定为直接去除模式，简化配置"
echo "  - 密度阈值调节：修改 FP_DENSITY_THRESHOLD \(0.1-0.8, 推荐0.3-0.5\)"
echo "  - 面积阈值调节：修改 FP_AREA_THRESHOLD \(推荐100000-200000像素\)"
echo "  - 评分阈值调节：修改 FP_SCORE_THRESHOLD \(1-5, 推荐2-4\)"
echo "  - 误报区域可视化：修改 SHOW_FALSE_POSITIVE \(true=显示紫色半透明mask, false=不显示\)"
echo "  - 使用专门的误报面积阈值和最小外接矩形密度，精准识别误报"
echo "  - 详细参数说明请参考 异物检测使用说明.md" 