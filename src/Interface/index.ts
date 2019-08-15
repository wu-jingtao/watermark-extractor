import * as fs from 'fs';
import * as path from 'path';
import * as canvas from 'canvas';
import * as _ from 'lodash';
import * as tf from '@tensorflow/tfjs';

const _stackSize = 5;
const _modelCache = new Map<string, tf.LayersModel>();

export type ImageDataType = string | Buffer | canvas.Image | canvas.Canvas;

/**
 * 转换图像数据。
 * 每张图片中水印的大小、颜色、位置都不能发生变化，每张图片的大小应当一致。
 * 可以传入多于5张图片，系统将自动选出其中最适合学习的图片
 */
export async function transformData(imageData: ImageDataType[]): Promise<tf.Tensor4D> {
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

        const result: Set<tf.Tensor3D> = new Set();

        for (let i = 0; result.size <= _stackSize; i++ , i %= 10) {
            if (r_rank[i].length > 0) result.add(r_rank[i].pop() as any);
            if (g_rank[i].length > 0) result.add(g_rank[i].pop() as any);
            if (b_rank[i].length > 0) result.add(b_rank[i].pop() as any);
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
 * 寻找水印在图片中的位置
 * @param imageData 图片数据
 * @param mode 以哪种模式寻找水印。colorful：彩色水印，white：白色水印，black：黑色水印
 * @returns 返回一个2维矩阵，值的大小表示存在水印的概率
 */
export async function findWatermarkPosition(imageData: tf.Tensor4D, mode: 'colorful' | 'white' | 'black'): Promise<tf.Tensor2D> {
    const _key = 'findWatermarkPosition_' + mode;
    if (_modelCache.has(_key))
        var model = _modelCache.get(_key) as tf.LayersModel;
    else {
        var model = await tf.loadLayersModel('file://' + path.join(__dirname, `../../bin/training_result/model/FindWatermarkPosition/${mode}/model.json`));
        _modelCache.set(_key, model);
    }

    return tf.tidy(() => {
        const result = model.predict(imageData.reshape([-1, 3 * _stackSize])) as tf.Tensor2D;
        return result.reshape([imageData.shape[0], imageData.shape[1]]) as tf.Tensor2D;
    });
}

/**
 * 将水印图片提取出来
 * @param imageData 图片数据
 * @param watermarkPosition 水印位置。注意传入的应当是一个bool类型的tensor
 * @param mode 以哪种模式提取水印。colorful：彩色水印，white：白色水印，black：黑色水印
 * @returns 返回提取出的水印图片。一个3维矩阵 [3, 3, 4]。注意Tensor中每个像素点的值都被除以了255
 */
export async function extractWatermark(imageData: tf.Tensor4D, watermarkPosition: tf.Tensor2D, mode: 'colorful' | 'white' | 'black'): Promise<tf.Tensor3D> {
    if (watermarkPosition.dtype !== 'bool') throw new Error('传入的 watermarkPosition 应当是bool类型的tensor');

    const _key = 'extractWatermark_' + mode;
    if (_modelCache.has(_key))
        var model = _modelCache.get(_key) as tf.LayersModel;
    else {
        var model = await tf.loadLayersModel('file://' + path.join(__dirname, `../../bin/training_result/model/ExtractWatermark/${mode}/model.json`));
        _modelCache.set(_key, model);
    }

    const position = await tf.whereAsync(watermarkPosition);
    return tf.tidy(() => {
        const pixels = tf.gatherND(imageData, position);
        const result = model.predict(pixels.reshape([-1, 3 * _stackSize])) as tf.Tensor2D;
        const reposition = tf.scatterND(position, result, [imageData.shape[0], imageData.shape[1], 4]) as tf.Tensor3D;

        position.dispose();
        return reposition;
    });
}

/**
 * 将tensor转换成PNG图片。
 * 对于水印位置tensor将会被转换成黑白图片，白色代表有水印。
 */
export async function tensorToPNG(data: tf.Tensor2D | tf.Tensor3D): Promise<Buffer> {
    const can = canvas.createCanvas(data.shape[1], data.shape[0]);
    const ctx = can.getContext('2d');

    ctx.putImageData(canvas.createImageData(await tf.browser.toPixels(data), data.shape[1], data.shape[0]), 0, 0);
    return can.toBuffer('image/png');
}