import * as fs from 'fs';
import * as path from 'path';
import * as _ from 'lodash';
import * as canvas from 'canvas';
import * as tf from '@tensorflow/tfjs';

/**
 * 训练数据提供器
 */
export class DataProvider {
    private static readonly _originalImageSize = [640, 360];    //原始图片大小
    private static readonly _watermarkImageSize = 150;          //水印图片大小
    private static readonly _originalImageDir = path.join(__dirname, '../../training_data/original');      //原始图片目录
    private static readonly _watermarkImageDir = path.join(__dirname, '../../training_data/watermark');    //水印图片目录

    private readonly _originalImagePaths: string[];     //原始图片路径
    private readonly _watermarkImagePaths: string[];    //水印图片路径
    private readonly _windowsSize: number;
    private readonly _stackSize: number;

    private readonly _originalImageCache: Map<number, tf.Tensor3D> = new Map();  //原始图片缓存，key对应_originalPaths的索引
    private readonly _watermarkImageCache: Map<number, tf.Tensor3D> = new Map(); //水印图片缓存

    /**
     * @param windowSize 要选取的图像方块大小。随机从图片上剪切`windowSize X windowSize`大小的图片给机器学习
     * @param stackSize 选取几张剪切下的图片进行堆叠(在RGB轴上进行)
     */
    constructor(windowSize: number, stackSize: number) {
        this._originalImagePaths = fs.readdirSync(DataProvider._originalImageDir).map(item => path.join(DataProvider._originalImageDir, item));
        this._watermarkImagePaths = fs.readdirSync(DataProvider._watermarkImageDir).map(item => path.join(DataProvider._watermarkImageDir, item));

        if (windowSize < 1) throw new Error('windowSize不可以小于1');
        if (stackSize < 1) throw new Error('stackSize不可以小于1');
        if (windowSize > DataProvider._watermarkImageSize) throw new Error('windowSize超过了水印图片的最大尺寸');
        if (stackSize > this._originalImagePaths.length) throw new Error('stackSize超过了数据集中original图片的数量');

        this._windowsSize = Math.trunc(windowSize);
        this._stackSize = Math.trunc(stackSize);
    }

    /**
     * 获取图片数据，返回图片的像素值被除以了255
     * @param type 原始图片还是水印图片
     * @param index 图片的索引号
     */
    private async _getImageData(type: 'original' | 'watermark', index: number): Promise<tf.Tensor3D> {
        const imagePaths = type === 'original' ? this._originalImagePaths : this._watermarkImagePaths;
        const imageCache = type === 'original' ? this._originalImageCache : this._watermarkImageCache;

        let result = imageCache.get(index);
        if (result === undefined) {
            const image = await canvas.loadImage(await fs.promises.readFile(imagePaths[index]));
            const tempCanvas = canvas.createCanvas(image.naturalWidth, image.naturalHeight);
            tempCanvas.getContext('2d', { alpha: false }).drawImage(image, 0, 0);
            result = tf.tidy(() => tf.browser.fromPixels(tempCanvas as any).div(255)) as tf.Tensor3D;
            imageCache.set(index, result);
        }

        return result;
    }

    /**
     * 准备数据，随机返回一张windowSize大小的水印图片和堆叠原始图片
     */
    private async _prepareData(): Promise<{ watermark: tf.Tensor3D, original: tf.Tensor4D, dispose: Function }> {
        const pickedOriginal = _.uniq(_.times(this._stackSize, () => Math.floor(this._originalImagePaths.length * Math.random())));
        const pickedWatermark = Math.floor(this._watermarkImagePaths.length * Math.random());
        if (pickedOriginal.length !== this._stackSize) return this._prepareData();   //如果因重复而没有挑选到足够的原始图片就重新挑选

        const watermark = await this._getImageData('watermark', pickedWatermark);
        const original = await Promise.all(pickedOriginal.map(item => this._getImageData('original', item)));

        //根据窗口大小，随机选择一个在图片上的剪切位置
        const cutPosition = {
            original: {
                x: Math.floor((DataProvider._originalImageSize[0] - this._windowsSize) * Math.random()),
                y: Math.floor((DataProvider._originalImageSize[1] - this._windowsSize) * Math.random())
            },
            watermark: {
                x: Math.floor((DataProvider._watermarkImageSize - this._windowsSize) * Math.random()),
                y: Math.floor((DataProvider._watermarkImageSize - this._windowsSize) * Math.random())
            }
        };

        //剪切水印图片
        const cutWatermark = watermark.slice([cutPosition.watermark.y, cutPosition.watermark.x], [this._windowsSize, this._windowsSize]);
        //剪切并堆叠原始图片
        const cutOriginal = tf.tidy(() => tf.stack(original.map(item => item.slice([cutPosition.original.y, cutPosition.original.x], [this._windowsSize, this._windowsSize])), 3)) as tf.Tensor4D;

        return {
            original: cutOriginal,
            watermark: cutWatermark,
            dispose() {
                cutWatermark.dispose();
                cutOriginal.dispose();
            }
        };
    }

    /**
     * 获取加过水印的训练数据
     */
    async getWaterMarkData() {
        const data = await this._prepareData();
        const randomAlpha = +Math.random().toFixed(2);  //随机水印透明度

        //混合原始图片与水印图片
        //公式为：原始图片*(1-透明度)+水印图片*透明度
        const mixed = tf.tidy(() => {
            const original = data.original.mul(1 - randomAlpha);
            const watermark = data.watermark.mul(randomAlpha).reshape([this._windowsSize, this._windowsSize, 3, 1]);
            return original.add(watermark);
        }) as tf.Tensor4D;

        //为水印图片添加alpha值
        const watermark = data.watermark.pad([[0, 0], [0, 0], [0, 1]], randomAlpha);

        data.dispose();
        return {
            test: mixed,
            answer: watermark,
            dispose() {
                watermark.dispose();
                mixed.dispose();
            }
        }
    }

    /**
     * 获取没有水印的训练数据
     */
    async getNoWatermarkData() {
        const data = await this._prepareData();
        const emptyWatermark = tf.zeros([this._windowsSize, this._windowsSize, 4]);
        data.watermark.dispose();
        return {
            test: data.original,
            answer: emptyWatermark,
            dispose() {
                data.dispose();
                emptyWatermark.dispose();
            }
        }
    }
}
