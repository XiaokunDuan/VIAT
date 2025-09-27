#!/bin/bash

# 指定要搜索的文件夹
DIRECTORIES=(
    # "asset"
    # "classifier"
    # "misc"
    # "models"
    # "run_train_nerf"
    "viat"
    # "benchmarking"
    # "datasets"
)

# 指定输出文件名
OUTPUT_FILE="collected_files.txt"

# 如果输出文件已存在，则清空它
> "$OUTPUT_FILE"

# 遍历指定的文件夹
for DIR in "${DIRECTORIES[@]}"; do
    # 检查文件夹是否存在
    if [ -d "$DIR" ]; then
        # 查找指定类型的文件并遍历它们
        find "$DIR" -type f \( -name "*.cpp" -o -name "*.py" -o -name "*.cu" -o -name "*.sh" -o -name "*.md" \) -print0 | while IFS= read -r -d $'\0' file; do
            # 将文件地址追加到输出文件
            echo "--- 文件地址: $file ---" >> "$OUTPUT_FILE"
            # 将文件内容追加到输出文件
            cat "$file" >> "$OUTPUT_FILE"
            # 在每个文件内容后添加一个换行符以分隔
            echo "" >> "$OUTPUT_FILE"
        done
    else
        echo "警告: 目录 '$DIR' 不存在，已跳过。"
    fi
done

echo "所有指定文件已成功聚合到 $OUTPUT_FILE"