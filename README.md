# watermark-extractor
通过神经网络将图片中的水印提取出来，为下一步去水印打下基础。

### Windows GPU开发环境搭建
1. 安装CUDA，[参考视频](https://www.youtube.com/watch?v=HExRhnO5Mqs)
    * 必须先安装visual studio, 安装vs时只需要安装c++模块即可。
    * 安装CUDA，安装前先看看[tfjs-node-gpu](https://github.com/tensorflow/tfjs-node#readme)支持的版本（必须完全一致，连小版本号都不许变）。安装时必须把翻墙打开，不然可能会失败。
    * 安装cuDNN，并配置环境变量
2. 安装tfjs-node-gpu
    * [windows版本常见错误](https://github.com/tensorflow/tfjs-node/blob/HEAD/WINDOWS_TROUBLESHOOTING.md)
    * 编译时如果遇到 `"gyp ERR! stack Error: C:\Program Files (x86)\MSBuild\14.0\bin\msbuild.exe failed with exit code: 1"` 错误，很有可能是javascript内存溢出造成的。添加一个环境变量 `NODE_OPTIONS=--max_old_space_size=4096` 即可。
    * 如果还是无法解决，可以通过以下方式查看以下编译出错的具体信息。
        ```
        Those logs have a truncated node-gyp log so it is hard to figure out what is going on.

        Clone the tfjs-node repo and running the following - it might give some more details:

        git clone https://github.com/tensorflow/tfjs-node.git
        cd tfjs-node
        npm install

        # This command probably fails:
        npm run enable-gpu

        # Get the logs from this:
        node-gyp rebuild
        If you don't have node-gyp - npm install -g node-gyp.
        ```
3. 安装 `tensorboard`。以超级用户运行 `pip install tensorboard`

### 训练数据准备
* 训练所使用的原始图片存放在 `training_data/original` 目录下。这里面的所有图片来自于[guanyuhan426的视频《东京印象 •春》](https://www.bilibili.com/video/av1084855/?p=2)，尺寸360p，通过 `ffmpeg -skip_frame nokey -i 视频名称.flv -vsync 0 -r 30 -f image2 %d.jpeg` 提取的关键帧。
* 训练使用的水印图片存放在 `training_data/watermark` 目录下。水印图片需要是不透明的，大小为150px。