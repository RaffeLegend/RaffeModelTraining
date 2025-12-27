#!/bin/bash

# 设置当前脚本所在的目录为工作目录（项目根目录）
cd "$(dirname "$0")"

# 激活虚拟环境（如果有）
# source /path/to/your/virtualenv/bin/activate

# 运行 Python 模块
python -m src.models.model_vlm.minicpm

# 检查 Python 命令是否成功执行
if [ $? -eq 0 ]; then
    echo "Execution successful!"
else
    echo "Execution failed!"
    exit 1
fi
