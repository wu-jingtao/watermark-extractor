# watermark-extractor
通过神经网络将图片中的水印提取出来，为下一步去水印打下基础。
> PS：提取的水印图片效果非常一般，实际使用时推荐多提取几张最后求个平均值。

### 使用方法
```javascript
import { transformData, tensorToPNG, findWatermarkPosition, extractWatermark } from 'watermark-extractor';
// import '@tensorflow/tfjs-node';      
// import '@tensorflow/tfjs-node-gpu';  //根据实际情况考虑是否开启GPU加速

// 将图片转换成Tensor。至少需要5张图片
const data = await transformData(_picturePaths);

// 找出水印在图片中的位置
const position = await findWatermarkPosition(data, '颜色模式');

// 选择概率大于70%的位置
const selected = tf.tidy(()=> position.greaterEqual(tf.fill(position.shape, 0.7)));

// 提取水印，PNG格式
const watermark = await extractWatermark(data, selected, '颜色模式');

//Tensor转换成图片
const png_position = tensorToPNG(position);
const png_watermark = tensorToPNG(watermark);

//清理内存
data.dispose();
position.dispose();
selected.dispose();
watermark.dispose();
```

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
* 从视频中提取关键帧：`ffmpeg -skip_frame nokey -i 视频名称.flv -vsync 0 -r 30 -f image2 %d.jpeg`。
* 注意通过关键帧来提取水印可能会导致水印图片变得模糊。