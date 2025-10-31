#!/bin/bash

# ==============================================================================
#  云端服务器环境快照脚本 (Cloud-Only Version)
# ==============================================================================
#
#  此脚本专门用于在云端服务器内部运行，以导出该服务器的
#  详细环境配置。
#
#  用法:
#  1. 在服务器上将此脚本保存为 'generate_cloud_snapshot.sh'
#  2. 给予执行权限: chmod +x generate_cloud_snapshot.sh
#  3. 运行脚本: ./generate_cloud_snapshot.sh
#  4. 检查生成的 'environment_snapshot_cloud.txt' 和 'viat_env.yml' 文件。
#  5. 将这两个文件下载到您的本地电脑并提交到 Git。
#
# ==============================================================================

# 定义输出文件名
SNAPSHOT_FILE="environment_snapshot_cloud.txt"
CONDA_ENV_FILE="viat_env.yml"

# --- [ Section 1: 导出 Conda 环境 (最佳恢复方式) ] ---
echo "正在导出 Conda 环境 'viat' 到 ${CONDA_ENV_FILE}..."
# 确保 conda 命令可用
source /root/miniconda3/etc/profile.d/conda.sh
conda activate viat
conda env export > "$CONDA_ENV_FILE"
echo "Conda 环境已成功导出。"
echo ""


# --- [ Section 2: 导出详细配置到文本文件 ] ---
echo "正在生成详细配置快照到 ${SNAPSHOT_FILE}..."

# 清空并初始化快照文件
echo "云端服务器环境快照 - 生成于: $(date)" > "$SNAPSHOT_FILE"
echo "======================================================================" >> "$SNAPSHOT_FILE"
echo "" >> "$SNAPSHOT_FILE"

# 2.1 系统和硬件信息
echo "--- [ 2.1 系统与硬件信息 ] ---" >> "$SNAPSHOT_FILE"
(
echo '>>> OS Version:'; cat /etc/os-release;
echo '';
echo '>>> Kernel Version:'; uname -a;
echo '';
echo '>>> CPU Info:'; lscpu;
echo '';
echo '>>> GPU & NVIDIA Driver Info:'; nvidia-smi;
) >> "$SNAPSHOT_FILE"
echo "" >> "$SNAPSHOT_FILE"

# 2.2 Shell 配置
echo "--- [ 2.2 Shell 配置文件 (~/.bashrc) ] ---" >> "$SNAPSHOT_FILE"
cat ~/.bashrc >> "$SNAPSHOT_FILE"
echo "" >> "$SNAPSHOT_FILE"

# 2.3 Conda 环境包列表 (作为参考)
echo "--- [ 2.3 Conda 环境 'viat' 包列表 (conda list) ] ---" >> "$SNAPSHOT_FILE"
conda list >> "$SNAPSHOT_FILE"
echo "" >> "$SNAPSHOT_FILE"

# 2.4 系统级安装包 (apt)
echo "--- [ 2.4 系统级已安装包 (apt list) ] ---" >> "$SNAPSHOT_FILE"
apt list --installed 2>/dev/null >> "$SNAPSHOT_FILE"
echo "" >> "$SNAPSHOT_FILE"

# 2.5 编译器版本
echo "--- [ 2.5 编译器版本 ] ---" >> "$SNAPSHOT_FILE"
(
echo '>>> GCC Version:'; gcc --version 2>/dev/null || echo "GCC not found.";
echo '';
echo '>>> G++ Version:'; g++ --version 2>/dev/null || echo "G++ not found.";
echo '';
echo '>>> NVCC (CUDA Compiler) Version:'; nvcc --version 2>/dev/null || echo "NVCC not found.";
) >> "$SNAPSHOT_FILE"
echo "" >> "$SNAPSHOT_FILE"

echo "======================================================================" >> "$SNAPSHOT_FILE"
echo "快照生成完毕！"
echo "1. 详细配置已保存到: ${SNAPSHOT_FILE}"
echo "2. Conda 环境配置已保存到: ${CONDA_ENV_FILE} (这是恢复Python环境的最佳方式)"
echo ""
echo "下一步: 请将这两个文件下载到您的本地电脑进行备份和版本控制。"