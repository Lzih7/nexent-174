#!/bin/bash

source venv/bin/activate
# 一键模型部署脚本：删除 -> 导入 -> 校验

set -e  # 遇到错误立即退出

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/models_config.json"
ENV_FILE="${SCRIPT_DIR}/.env.txt"
API_URL="http://localhost:5010"

echo "🚀 Nexent 一键模型部署脚本"
echo "配置文件: $CONFIG_FILE"
echo "环境变量文件: $ENV_FILE"
echo "API地址: $API_URL"
echo

# 加载环境变量
if [ -f "$ENV_FILE" ]; then
    echo "📋 加载环境变量..."
    set -a  # 自动导出所有变量
    source "$ENV_FILE"
    set +a  # 关闭自动导出
    echo "✅ 环境变量加载完成"
    echo
else
    echo "⚠️  警告: 环境变量文件不存在: $ENV_FILE"
    echo "   某些模型可能无法正常工作"
    echo
fi

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 步骤1: 删除所有现有模型
echo "🗑️  步骤1: 删除所有现有模型..."
python "$SCRIPT_DIR/nexent_model_manager.py" --base-url "$API_URL" --delete-all
if [ $? -ne 0 ]; then
    echo "❌ 删除模型失败"
    exit 1
fi
echo "✅ 删除完成"
echo

# 步骤2: 从配置文件导入模型
echo "📥 步骤2: 从配置文件导入模型..."
python "$SCRIPT_DIR/nexent_model_manager.py" --base-url "$API_URL" --config "$CONFIG_FILE"
if [ $? -ne 0 ]; then
    echo "❌ 导入模型失败"
    exit 1
fi
echo "✅ 导入完成"
echo

# 步骤3: 校验所有模型连通性
echo "🔍 步骤3: 校验所有模型连通性..."
python "$SCRIPT_DIR/nexent_model_manager.py" --base-url "$API_URL" --verify-all
if [ $? -ne 0 ]; then
    echo "⚠️  部分模型连通性异常，但部署流程已完成"
    exit 0
fi

echo
echo "🎉 一键部署完成！所有模型已成功导入并验证连通性正常"