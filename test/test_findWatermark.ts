import * as path from 'path';
import * as fs from 'fs-extra';
import * as canvas from 'canvas';
import * as tf from '@tensorflow/tfjs-node-gpu';
import { findWatermark, transformData } from '../src/Interface';

const _testingDataDir = path.join(__dirname, '../testing_data');
const _pictures = fs.readdirSync(_testingDataDir).map(item => path.join(_testingDataDir, item));

async function test_findWatermark() {
    const savePath = path.join(__dirname, '../bin/testing_result/findWatermark.png');
    await fs.ensureFile(savePath);
    await fs.remove(savePath);

    const data = await transformData(_pictures);
    const result = await findWatermark(data);
    const can = canvas.createCanvas(result.shape[1], result.shape[0]);
    const ctx = can.getContext('2d');
    ctx.putImageData(canvas.createImageData(await tf.browser.toPixels(result.logicalNot().mul(255).toInt() as tf.Tensor2D), result.shape[1], result.shape[0]), 0, 0);
    await fs.writeFile(savePath, can.toBuffer('image/png'));
    console.log('创建完成：findWatermark.png');
}

test_findWatermark().catch(console.error);