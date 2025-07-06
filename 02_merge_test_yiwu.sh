#!/bin/bash

# =================================================================
# 催化剂异物异形检测脚本启动器 (merge_test_yiwu.py)
# 一键启动催化剂异物异形检测功能
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
OUTPUT_DIR="./data/catalyst_merge/vis_result_yiwu_v0001"

# 异物检测参数
MIN_COMPONENT_AREA=500 # 连通域预过滤最小面积阈值
MIN_AREA=0             # 最小连通域面积阈值
MAX_AREA=50000         # 最大连通域面积阈值
MIN_ASPECT_RATIO=1.5   # 最小长宽比阈值
MAX_ASPECT_RATIO=20.0  # 最大长宽比阈值
MIN_SOLIDITY=0.6       # 最小实心度阈值
EDGE_THRESHOLD=50      # 边缘区域阈值(像素)

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
echo "  最小面积阈值: $MIN_AREA"
echo "  最大面积阈值: $MAX_AREA"
echo "  长宽比范围: $MIN_ASPECT_RATIO - $MAX_ASPECT_RATIO"
echo "  最小实心度: $MIN_SOLIDITY"
echo "  边缘阈值: ${EDGE_THRESHOLD}px"
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
    --min-area $MIN_AREA \
    --max-area $MAX_AREA \
    --min-component-area $MIN_COMPONENT_AREA \
    --min-aspect-ratio $MIN_ASPECT_RATIO \
    --max-aspect-ratio $MAX_ASPECT_RATIO \
    --min-solidity $MIN_SOLIDITY \
    --edge-threshold $EDGE_THRESHOLD \
    --image-exts $IMAGE_EXTENSIONS"

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
echo ""
echo "📊 可调节参数："
echo "  - 如需调整检测灵敏度，请修改脚本中的检测参数"
echo "  - 详细参数说明请参考 异物检测使用说明.md" 