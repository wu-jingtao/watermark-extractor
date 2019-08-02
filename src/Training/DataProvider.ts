import * as fs from 'fs';
import * as path from 'path';
import * as _ from 'lodash';
import * as canvas from 'canvas';
import * as tf from '@tensorflow/tfjs';

/**
 * 训练数据提供器
 */
export class DataProvider {
    private static readonly _imageSize = [640, 360]; //训练图片大小
    private static readonly _imageDir = path.join(__dirname, '../../training_data'); //训练图片目录
    private static readonly _imagePaths: string[] = fs.readdirSync(DataProvider._imageDir).map(item => path.join(DataProvider._imageDir, item)); //训练图片路径

    private readonly _stackSize: number;
    private readonly _minTransparency: number;
    private readonly _allowDuplicate: boolean;

    private readonly _imageCache: Map<number, tf.Tensor3D> = new Map();  //训练图片缓存，key对应_imagePaths的索引

    /**
     * @param stackSize 选取几张图片进行堆叠(在RGB轴上进行)
     * @param minTransparency 最小透明度
     * @param allowDuplicate 是否允许选取训练图片的时候存在重复
     */
    constructor(stackSize: number, minTransparency: number, allowDuplicate: boolean) {
        if (stackSize < 1) throw new Error('stackSize不可以小于1');
        if (stackSize > DataProvider._imagePaths.length) throw new Error('stackSize超过了训练图片的数量');
        if (minTransparency > 1 || minTransparency < 0) throw new Error('minTransparency取值必须在0-1之间');

        this._stackSize = Math.trunc(stackSize);
        this._minTransparency = minTransparency;
        this._allowDuplicate = allowDuplicate;
    }

    /**
     * 获取图片数据，返回图片的像素值被除以了255
     * @param index 图片的索引号
     */
    private async _getImageData(index: number): Promise<tf.Tensor3D> {
        let result = this._imageCache.get(index);

        if (result === undefined) {
            const image = await canvas.loadImage(await fs.promises.readFile(DataProvider._imagePaths[index]));
            const tempCanvas = canvas.createCanvas(DataProvider._imageSize[0], DataProvider._imageSize[1]);
            tempCanvas.getContext('2d', { alpha: false }).drawImage(image, 0, 0);
            result = tf.tidy(() => tf.browser.fromPixels(tempCanvas as any).div(255)) as tf.Tensor3D;
            this._imageCache.set(index, result);
        }

        return result;
    }

    /**
     * 准备数据，生成水印和堆叠训练图片
     */
    private async _prepareData(): Promise<{ watermark: tf.Tensor1D, image: tf.Tensor2D, dispose: Function }> {
        if (this._allowDuplicate)
            var pickedImage = _.times(this._stackSize, () => Math.floor(DataProvider._imagePaths.length * Math.random()));
        else {
            var pickedImage = _.uniq(_.times(this._stackSize * 2, () => Math.floor(DataProvider._imagePaths.length * Math.random())));
            if (pickedImage.length < this._stackSize) return this._prepareData();   //如果因重复而没有挑选到足够的训练图片就重新挑选
            pickedImage.length = this._stackSize;
        }

        const watermark = tf.tensor1d(_.times(3, Math.random)); //随机选择一个颜色作为水印
        const images = await Promise.all(pickedImage.map(item => this._getImageData(item)));

        //随机选择一个在图片上的剪切位置
        const cutPosition_x = Math.floor((DataProvider._imageSize[0] - 1) * Math.random());
        const cutPosition_y = Math.floor((DataProvider._imageSize[1] - 1) * Math.random());

        //剪切并堆叠训练图片
        const cutImage = tf.tidy(() => tf.stack(images.map(item => item.slice([cutPosition_y, cutPosition_x], [1, 1])), 3).squeeze()) as tf.Tensor2D;

        return {
            image: cutImage,
            watermark: watermark,
            dispose() {
                watermark.dispose();
                cutImage.dispose();
            }
        };
    }

    /**
     * 获取加过水印的训练数据
     * @param pieces 要获取多少数据
     */
    async getWaterMarkData(pieces = 1) {
        const result: { test: tf.Tensor2D, answer: tf.Tensor1D, dispose: Function }[] = [];

        for (let i = 0; i < pieces; i++) {
            const data = await this._prepareData();
            const randomAlpha = +(this._minTransparency + (1 - this._minTransparency) * Math.random()).toFixed(2);  //随机水印透明度

            //混合训练图片与水印图片
            //公式为：训练图片*(1-透明度)+水印图片*透明度
            const mixed = tf.tidy(() => {
                const original = data.image.mul(1 - randomAlpha);
                const watermark = data.watermark.mul(randomAlpha).reshape([3, 1]);
                return original.add(watermark);
            }) as tf.Tensor2D;

            //为水印图片添加alpha值
            const watermark = data.watermark.pad([[0, 1]], randomAlpha);

            data.dispose();
            result.push({
                test: mixed,
                answer: watermark,
                dispose() {
                    watermark.dispose();
                    mixed.dispose();
                }
            });
        }

        return result;
    }

    /**
     * 获取没有水印的训练数据
     * @param pieces 要获取多少数据
     */
    async getNoWatermarkData(pieces = 1) {
        const result: { test: tf.Tensor2D, answer: tf.Tensor1D, dispose: Function }[] = [];

        for (let i = 0; i < pieces; i++) {
            const data = await this._prepareData();
            const emptyWatermark: tf.Tensor1D = tf.zeros([4]);
            data.watermark.dispose();
            result.push({
                test: data.image,
                answer: emptyWatermark,
                dispose() {
                    data.dispose();
                    emptyWatermark.dispose();
                }
            });
        }

        return result;
    }

    /**
     * 清空所有图片缓存，释放内存资源
     */
    dispose() {
        this._imageCache.forEach(item => item.dispose());
    }
}
