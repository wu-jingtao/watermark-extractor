import * as _ from 'lodash';
import * as tf from '@tensorflow/tfjs';

/**
 * 训练数据提供器
 */
export class DataProvider {

    private readonly _stackSize: number;
    private readonly _minTransparency: number;

    /**
     * @param stackSize 选取几张图片进行堆叠(在RGB轴上进行)
     * @param minTransparency 最小透明度
     */
    constructor(stackSize: number, minTransparency: number) {
        if (stackSize < 1) throw new Error('stackSize不可以小于1');
        if (minTransparency > 1 || minTransparency < 0) throw new Error('minTransparency取值必须在0-1之间');

        this._stackSize = Math.trunc(stackSize);
        this._minTransparency = minTransparency;
    }

    /**
     * 随机颜色值
     */
    private _randomColorValue() {
        return Math.round(255 * Math.sqrt(Math.random() * Math.random())) / 255;
    }

    /**
     * 随机透明度
     */
    private _randomAlphaValue() {
        const alpha = this._minTransparency + (1 - this._minTransparency) * Math.sqrt(Math.random() * Math.random());
        return Math.round(255 * alpha) / 255;
    }

    /**
     * 堆叠彩色像素
     */
    get colorPixelStack() {
        return tf.tensor2d(_.times(this._stackSize * 3, this._randomColorValue), [3, this._stackSize]);
    }

    /**
     * 彩色像素
     */
    get colorPixel() {
        return tf.tensor1d(_.times(3, this._randomColorValue));
    }

    /**
     * 白色像素
     */
    get whitePixel() {
        return tf.ones([3]) as tf.Tensor1D;
    }

    /**
     * 黑色像素
     */
    get blackPixel() {
        return tf.zeros([3]) as tf.Tensor1D;
    }

    /**
     * 以随机透明度混合原始像素与水印像素
     * @return 返回 mixed：混合了水印的像素点，watermark：添加了alpha通道的水印图片
     */
    mixer(original: tf.Tensor2D, watermark: tf.Tensor1D): { mixed: tf.Tensor2D, watermark: tf.Tensor1D } {
        const randomAlpha = this._randomAlphaValue();
        return {
            mixed: tf.tidy(() => original.mul(1 - randomAlpha).add(watermark.mul(randomAlpha).reshape([3, 1]))),
            watermark: watermark.pad([[0, 1]], randomAlpha)
        };
    }
}
