#!/bin/bash
# 当需要完全重新部署nexent,包括代码和数据时,使用此脚本.
# 此脚本会停止并删除所有Nexent相关的容器和数据目录,然后清理Docker资源.

# 检查是否为root用户
if [ "$(id -u)" -ne 0 ]; then
    echo "请使用root权限运行此脚本."
    exit 1
fi

echo "🧹 开始清理Nexent数据..."

# 进入docker目录
cd ../docker

# 方法1：尝试正常停止
echo "🛑 尝试正常停止服务..."
docker compose down -v 2>/dev/null || {
    echo "⚠️ 正常停止失败，使用强制停止..."
    
    # 方法2：强制停止所有Nexent相关容器
    echo "🔨 强制停止Nexent容器..."
    docker stop nexent-elasticsearch nexent-postgresql nexent nexent-web nexent-data-process nexent-redis nexent-minio nexent-openssh-server 2>/dev/null || true
    
    # 删除容器
    echo "🗑️ 删除Nexent容器..."
    docker rm nexent-elasticsearch nexent-postgresql nexent nexent-web nexent-data-process nexent-redis nexent-minio nexent-openssh-server 2>/dev/null || true
    
    # 如果还有其他容器在运行，全部停止
    if [ "$(docker ps -q)" ]; then
        echo "🛑 停止所有剩余容器..."
        docker stop $(docker ps -q)
        docker rm $(docker ps -aq)
    fi
}

# 删除数据目录
echo "🗑️ 删除数据目录..."
sudo rm -rf ~/nexent-data
rm -rf ~/nexent

# 清理Docker资源
echo "🐳 清理Docker资源..."
docker container prune -f
docker volume prune -f

# 删除Nexent相关的镜像（可选）,这里如如删除,重新build会花费大量的时间.
# echo "🖼️ 清理Nexent镜像..."
# docker images | grep nexent | awk '{print $3}' | xargs -r docker rmi -f 2>/dev/null || true

echo "✅ 清理完成！现在可以重新部署系统了。"