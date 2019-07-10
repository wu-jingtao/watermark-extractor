import * as fs from 'fs';
import * as path from 'path';
import * as _ from 'lodash';
import * as canvas from 'canvas';
import * as tf from '@tensorflow/tfjs';

/**
 * 训练数据提供器
 */
export class DataProvider {

    private readonly _originalPaths: string[];     //原始图片路径
    private readonly _watermarkPaths: string[];    //水印图片路径
    private readonly _windowsSize: number;
    private readonly _stackSize: number;

    private readonly _originalImageCache: Map<number, tf.Tensor3D> = new Map();  //原始图片缓存，key对应_originalPaths的索引
    private readonly _watermarkImageCache: Map<number, tf.Tensor3D> = new Map(); //水印图片缓存，水印包含alpha通道

    /**
     * @param dataSet 使用训练集还是测试集的数据
     * @param windowSize 要选取的图像方块大小。随机从图片上剪切`windowSize X windowSize`大小的图片给机器学习
     * @param stackSize 选取几张剪切下的图片进行垂直堆叠
     */
    constructor(dataSet: 'training' | 'testing', windowSize: number, stackSize: number) {
        const dataPath = path.join(__dirname, '../../', dataSet === 'training' ? 'training_data' : 'testing_data');
        this._originalPaths = fs.readdirSync(path.join(dataPath, 'original')).map(item => path.join(dataPath, 'original', item));
        this._watermarkPaths = fs.readdirSync(path.join(dataPath, 'watermark')).map(item => path.join(dataPath, 'watermark', item));

        if (windowSize < 1) throw new Error('windowSize不可以小于1');
        if (stackSize < 1) throw new Error('stackSize不可以小于1');
        if (stackSize > this._originalPaths.length) throw new Error('stackSize超过了数据集中original图片的数量');

        this._windowsSize = Math.trunc(windowSize);
        this._stackSize = Math.trunc(stackSize);
    }

    /**
     * 准备数据，随机返回一张windowSize大小的水印图片和stackSize张原始图片
     */
    private async _prepareData(): Promise<{ watermark: tf.Tensor3D, original: tf.Tensor3D[], dispose: Function }> {
        const pickedOriginal = _.uniq(_.times(this._stackSize, () => Math.floor(this._originalPaths.length * Math.random())));
        const pickedWatermark = Math.floor(this._watermarkPaths.length * Math.random());

        if (pickedOriginal.length !== this._stackSize) return this._prepareData();   //如果因重复而没有挑选到足够的原始图片就重新挑选

        let watermark = this._watermarkImageCache.get(pickedWatermark);
        if (watermark === undefined) {
            watermark = tf.browser.fromPixels(await canvas.loadImage(this._watermarkPaths[pickedWatermark]) as any, 4);
            this._watermarkImageCache.set(pickedWatermark, watermark);
        }

        let original = [];
        for (const item of pickedOriginal) {
            let ori = this._originalImageCache.get(item);
            if (ori === undefined) {
                ori = tf.browser.fromPixels(await canvas.loadImage(this._originalPaths[item]) as any);
                this._originalImageCache.set(item, ori);
            }
            original.push(ori);
        }

        //根据窗口大小，随机选择一个在图片上的剪切位置
        const cutPosition = {
            original: {
                x: Math.floor((1920 - this._windowsSize) * Math.random()),
                y: Math.floor((1080 - this._windowsSize) * Math.random())
            },
            watermark: {
                x: Math.floor((100 - this._windowsSize) * Math.random()),
                y: Math.floor((100 - this._windowsSize) * Math.random())
            }
        };

        const cutWatermark = watermark.slice([cutPosition.watermark.x, cutPosition.watermark.y], [this._windowsSize, this._windowsSize]);
        const cutOriginal = original.map(item => item.slice([cutPosition.original.x, cutPosition.original.y], [this._windowsSize, this._windowsSize]))

        return {
            original: cutOriginal,
            watermark: cutWatermark,
            dispose() {
                cutWatermark.dispose();
                cutOriginal.forEach(item => item.dispose());
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
            const watermarkAlpha = data.watermark.stridedSlice([0, 0, 3], [this._windowsSize, this._windowsSize, 4], [1, 1, 1]).mul(randomAlpha);   //水印透明度
            const watermarkColor = data.watermark.stridedSlice([0, 0, 0], [this._windowsSize, this._windowsSize, 3], [1, 1, 1]);
            const originalAlpha = tf.onesLike(watermarkAlpha).sub(watermarkAlpha);  //原始图片透明度
            return data.original.map(item => item.mul(originalAlpha).add(watermarkColor.mul(watermarkAlpha)));
        });

        //调整水印图片的透明度
        const watermark = tf.tidy(() => data.watermark.mul(tf.tensor([1, 1, 1, randomAlpha], [1, 1, 4])));

        data.dispose();

        return {
            test: mixed,
            answer: watermark,
            dispose() {
                watermark.dispose();
                mixed.forEach(item => item.dispose());
            }
        }
    }

    /**
     * 获取没有水印的训练数据
     */
    async getNoWatermarkData() {
        const data = await this._prepareData();
        const emptyWatermark = tf.zerosLike(data.watermark);
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
