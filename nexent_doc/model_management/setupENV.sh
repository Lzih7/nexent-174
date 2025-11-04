#!/bin/bash

# 简单的虚拟环境设置脚本
echo "正在设置Python虚拟环境..."

# 删除现有虚拟环境（如果存在）
if [ -d "venv" ]; then
    echo "删除现有虚拟环境..."
    rm -rf venv
fi

# 创建新的虚拟环境
echo "创建新的虚拟环境..."
python3.11 -m venv venv

# 激活虚拟环境并安装依赖
echo "激活虚拟环境并安装依赖..."
source venv/bin/activate
pip install --upgrade pip
pip install requests

echo "虚拟环境设置完成！"
echo "使用方法："
echo "  激活环境: source venv/bin/activate"
echo "  运行脚本: python nexent_model_manager.py --help"
echo "  退出环境: deactivate"