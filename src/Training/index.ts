import * as fs from 'fs';
import * as path from 'path';
import * as canvas from 'canvas';
import * as tfjs from '@tensorflow/tfjs';
import { DataProvider } from "./DataProvider";

(async () => {
    const test = new DataProvider('testing', 50, 1);
    const data = await test.getWaterMarkData();

    const temp = canvas.createCanvas(50, 50);
    const ctx = temp.getContext('2d');

    ctx.putImageData(canvas.createImageData(await tfjs.browser.toPixels(data.test[0]), 50, 50), 0, 0);
    const original = temp.toBuffer('image/png');

    ctx.clearRect(0, 0, 50, 50);
    ctx.putImageData(canvas.createImageData(await tfjs.browser.toPixels(data.answer), 50, 50), 0, 0);
    const watermark = temp.toBuffer('image/png');

    fs.writeFileSync(path.join(__dirname, '../../bin', 'original.png'), original);
    fs.writeFileSync(path.join(__dirname, '../../bin', 'watermark.png'), watermark);
    console.log('ok');
})()
