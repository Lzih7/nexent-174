#!/bin/bash

# 我们修改的代码在Step-by-Step/tool_collection 目录下.

clear
echo "复制工具集到后端目录"
cp -R tool_collection/ ../backend/tool_collection/

echo "更新后端的pyproject.toml支持pypdf"
cp pyproject.toml ../backend/


cd ..
pwd


# 1. 构建自定义镜像
docker stop nexent


# build 后端代码,我们写的工具都在后端代码中.
docker rmi -f nexent/nexent:latest
docker build  --progress=plain -t nexent/nexent -f make/main/Dockerfile .

# 如果前后端的代码不匹配就需要重新构建前端镜像(放开下面一句),否则不需要.
# docker rmi -f nexent/nexent-web:latest
# docker build --progress=plain -t nexent/nexent-web -f make/web/Dockerfile .

# docker rmi -f nexent/nexent-data-process:latest
# docker build --progress=plain -t nexent/nexent-data-process -f make/data_process/Dockerfile .


# 2. 配置环境变量
cd docker
cp .env.example .env
echo "NEXENT_IMAGE=nexent/nexent:latest" >> .env
echo "NEXENT_WEB_IMAGE=nexent/nexent-web:latest" >> .env
echo "NEXENT_DATA_PROCESS_IMAGE=nexent/nexent-data-process:latest" >> .env


# 3. 部署服务 - 使用环境变量方式避免所有交互提示
./deploy.sh --mode 1 --version 1 --is-mainland N --enable-terminal N --root-dir "$HOME/nexent-data"
docker images | grep nexent