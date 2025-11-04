## 编写mcp工具

nexent 的mcp工具是在根目录的backend/tool_collection/下面.
为了防止从git pull 代码时会覆盖掉我们编写的工具,
我们自己编写的工具放在当前目录的 tool_collection/中.


我们提供了编译脚本([build-and-deploy.sh](build-and-deploy.sh)),这个脚本会自动把自己编写的工具集复制到根目录的后端目录,然后执行docker build命令,生成新的镜像.


## 编译新镜像生成新容器

执行脚本

```bash
#进入Step-by-Step目录
chmod +x build-and-deploy.sh
./build-and-deploy.sh
```