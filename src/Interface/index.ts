import * as fs from 'fs';
import * as path from 'path';
import * as canvas from 'canvas';
import * as _ from 'lodash';
import * as tf from '@tensorflow/tfjs';

type ImageDataType = string | Buffer | canvas.Image | canvas.Canvas;

const _stackSize = 20;
let _model_predictHasWatermark: tf.LayersModel;
let _model_extractWatermark: tf.LayersModel;

/**
 * 转换图像数据。可以传入多于_stackSize张图片，系统将自动选出其中最适合学习的图片
 */
async function _transformData(imageData: ImageDataType[]): Promise<tf.Tensor4D> {
    if (imageData.length < _stackSize) throw new Error(`至少传入${_stackSize}张图片`);

    const images: tf.Tensor3D[] = [];

    let lastWidth: number | undefined = undefined;
    let lastHeight: number | undefined = undefined;

    for (let item of imageData) {
        if (typeof item === 'string')
            item = await fs.promises.readFile(item);

        if (Buffer.isBuffer(item))
            item = await canvas.loadImage(item);

        if (item instanceof canvas.Image) {
            const can = canvas.createCanvas(item.naturalWidth, item.naturalHeight);
            can.getContext('2d', { alpha: false }).drawImage(item, 0, 0);
            item = can;
        }

        if (lastWidth === undefined)
            lastWidth = item.width;
        else if (lastWidth !== item.width)
            throw new Error('传入图片的宽度不一致');

        if (lastHeight === undefined)
            lastHeight = item.height;
        else if (lastHeight !== item.height)
            throw new Error('传入图片的高度不一致');

        images.push(tf.tidy(() => tf.browser.fromPixels(item as any).div(255)) as tf.Tensor3D);
    }

    if (imageData.length === _stackSize)
        var selected = images;
    else {
        const r_rank: tf.Tensor3D[][] = _.times(10, () => []);  
        const g_rank: tf.Tensor3D[][] = _.times(10, () => []);
        const b_rank: tf.Tensor3D[][] = _.times(10, () => []);

        for (const item of images) {
            //将图片的每个颜色通道提出来，然后按照0-9排序
            const [r, g, b] = tf.tidy(() => item.split(3, 2).map(item => item.mean().mul(10).floor()));
            r_rank[(await r.data())[0]].push(item); r.dispose();
            g_rank[(await g.data())[0]].push(item); g.dispose();
            b_rank[(await b.data())[0]].push(item); b.dispose();
        }

        r_rank.sort((a, b) => a.length - b.length);
        g_rank.sort((a, b) => a.length - b.length);
        b_rank.sort((a, b) => a.length - b.length);

        const result: Set<tf.Tensor3D> = new Set();

        for (let i = 0; i < r_rank.length && result.size < _stackSize; i++) {
            r_rank[i].forEach(item => result.add(item));
            g_rank[i].forEach(item => result.add(item));
            b_rank[i].forEach(item => result.add(item));
        }

        var selected = [...result];
        selected.length = _stackSize;
    }

    return tf.tidy(() => {
        const data = tf.stack(_.shuffle(selected), 3);
        images.map(item => item.dispose());
        return data;
    }) as tf.Tensor4D;
}

/**
 * 寻找图片中的水印
 * @param imageData 被加过水印的图片。每张图片中水印的大小、颜色、位置都不能发生变化。每张图片的大小应当一致。
 * @returns 返回一个2维矩阵，true表示该像素点有水印，false表示没有
 */
export async function findWatermark(imageData: ImageDataType[] | tf.Tensor4D) {
    if (_model_predictHasWatermark === undefined)
        _model_predictHasWatermark = await tf.loadLayersModel('file://' + path.join(__dirname, '../../bin/training_result/model/PredictHasWatermark/model.json'));

    const data = Array.isArray(imageData) ? await _transformData(imageData) : imageData;

    return tf.tidy(() => {
        const result = _model_predictHasWatermark.predict(data.reshape([-1, 3 * _stackSize])) as tf.Tensor2D;
        if (Array.isArray(imageData)) data.dispose();
        return result.round().toBool().reshape([data.shape[0], data.shape[1]]) as tf.Tensor2D;
    });
}

/**
 * 将水印图片提取出来
 * @param imageData 被加过水印的图片。每张图片中水印的大小、颜色、位置都不能发生变化。每张图片的大小应当一致。
 * @param returnTensor 以png格式图片返回。
 */
export async function extractWatermark(imageData: ImageDataType[], returnTensor?: false): Promise<Buffer>
/**
 * 将水印图片提取出来
 * @param imageData 被加过水印的图片。每张图片中水印的大小、颜色、位置都不能发生变化。每张图片的大小应当一致。
 * @param returnTensor 以Tensor的形式返回。注意Tensor中每个像素点的值都被除以了255
 * @returns 返回提取出的水印图片。一个3维矩阵 [3, 3, 4]
 */
export async function extractWatermark(imageData: ImageDataType[], returnTensor: true): Promise<tf.Tensor3D>
export async function extractWatermark(imageData: ImageDataType[], returnTensor: boolean = false) {
    if (_model_extractWatermark === undefined)
        _model_extractWatermark = await tf.loadLayersModel('file://' + path.join(__dirname, '../../bin/training_result/model/ExtractWatermark/model.json'));

    const data = await _transformData(imageData);
    const hasWatermarkPosition = await findWatermark(data);
    const waterPosition = await tf.whereAsync(hasWatermarkPosition);

    const result = tf.tidy(() => {
        const pixels = tf.gatherND(data, waterPosition);
        const result = _model_extractWatermark.predict(pixels.reshape([-1, 3 * _stackSize])) as tf.Tensor2D;
        const reposition = tf.scatterND(waterPosition, result, [data.shape[0], data.shape[1], 4]) as tf.Tensor3D;

        data.dispose();
        waterPosition.dispose();
        hasWatermarkPosition.dispose();

        return reposition;
    });

    if (returnTensor)
        return result;
    else {
        const can = canvas.createCanvas(data.shape[1], data.shape[0]);
        const ctx = can.getContext('2d');

        const picture = tf.tidy(() => result.mul(255).round().toInt()) as tf.Tensor3D;
        ctx.putImageData(canvas.createImageData(await tf.browser.toPixels(picture), data.shape[1], data.shape[0]), 0, 0);

        result.dispose();
        picture.dispose();
        return can.toBuffer('image/png');
    }
}