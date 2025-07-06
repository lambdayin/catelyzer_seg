#!/bin/bash

# =================================================================
# 催化剂长度测量脚本启动器 (merge_test.py)
# 一键启动催化剂分割和长度测量功能
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
OUTPUT_DIR="./data/catalyst_merge/vis_result"

# 网格标定参数
GRID_LENGTH=36      # 网格物理长度 (毫米)
GRID_PIXEL=651      # 网格像素距离

# 支持的图像格式
IMAGE_EXTENSIONS="jpg jpeg png bmp tiff"

# GPU设备 (设置为空字符串使用CPU)
CUDA_DEVICE="0"

# =================================================================
# 预检查
# =================================================================

print_info "开始催化剂长度测量任务..."
print_info "=================================="

# 检查Python环境
if ! command -v python &> /dev/null; then
    print_error "Python未找到，请确保Python已安装并在PATH中"
    exit 1
fi

# 检查脚本文件
if [ ! -f "merge_test.py" ]; then
    print_error "merge_test.py文件未找到"
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
echo "  网格长度: ${GRID_LENGTH}mm"
echo "  网格像素: ${GRID_PIXEL}px"
echo "  GPU设备: ${CUDA_DEVICE:-CPU模式}"
echo ""

# 询问是否继续
read -p "按回车键开始处理，或输入 'q' 退出: " confirm
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

print_info "开始处理图像..."
echo "=================================="

# 构建Python命令
python_cmd="python merge_test_bak.py \"$MODEL_PATH\" \
    --input-dir \"$INPUT_DIR\" \
    --output-dir \"$OUTPUT_DIR\" \
    --grid-len $GRID_LENGTH \
    --grid-pixel $GRID_PIXEL \
    --image-exts $IMAGE_EXTENSIONS"

# 记录开始时间
start_time=$(date +%s)

# 执行命令
if eval $python_cmd; then
    # 计算耗时
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    print_success "处理完成！"
    echo "=================================="
    print_info "总耗时: ${duration}秒"
    print_info "结果保存在: $OUTPUT_DIR"
    
    # 统计输出文件
    result_count=$(find "$OUTPUT_DIR" -name "*_result.*" | wc -l)
    print_info "生成结果图像: $result_count 张"

    
else
    print_error "处理失败！"
    exit 1
fi

print_success "脚本执行完成！" 