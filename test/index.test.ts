import * as path from 'path';
import * as fs from 'fs-extra';
import '@tensorflow/tfjs-node';

import { transformData, tensorToPNG, findWatermarkPosition, extractWatermark } from '../src/Interface';

const testingResultDir = path.join(__dirname, '../bin/testing_result');
fs.ensureDirSync(testingResultDir);

it('测试 - 彩色水印', async function () {
    const result_position = path.join(testingResultDir, 'colorful_position.png');
    const result_watermark = path.join(testingResultDir, 'colorful_watermark.png');
    await fs.remove(result_position);
    await fs.remove(result_watermark);

    const testingDataDir = path.join(__dirname, './testing_data/colorful');
    const testingFiles = (await fs.readdir(testingDataDir)).map(item => path.join(testingDataDir, item));

    const data = await transformData(testingFiles);
    const position = await findWatermarkPosition(data, 'colorful');
    const watermark = await extractWatermark(data, position, 'colorful');

    await fs.writeFile(result_position, await tensorToPNG(position));
    await fs.writeFile(result_watermark, await tensorToPNG(watermark));
});

it('测试 - 白色水印', async function () {
    const result_position = path.join(testingResultDir, 'white_position.png');
    const result_watermark = path.join(testingResultDir, 'white_watermark.png');
    await fs.remove(result_position);
    await fs.remove(result_watermark);

    const testingDataDir = path.join(__dirname, './testing_data/white');
    const testingFiles = (await fs.readdir(testingDataDir)).map(item => path.join(testingDataDir, item));

    const data = await transformData(testingFiles);
    const position = await findWatermarkPosition(data, 'white');
    const watermark = await extractWatermark(data, position, 'white');

    await fs.writeFile(result_position, await tensorToPNG(position));
    await fs.writeFile(result_watermark, await tensorToPNG(watermark));
});

it('测试 - 黑色水印', async function () {
    const result_position = path.join(testingResultDir, 'black_position.png');
    const result_watermark = path.join(testingResultDir, 'black_watermark.png');
    await fs.remove(result_position);
    await fs.remove(result_watermark);

    const testingDataDir = path.join(__dirname, './testing_data/black');
    const testingFiles = (await fs.readdir(testingDataDir)).map(item => path.join(testingDataDir, item));

    const data = await transformData(testingFiles);
    const position = await findWatermarkPosition(data, 'black');
    const watermark = await extractWatermark(data, position, 'black');

    await fs.writeFile(result_position, await tensorToPNG(position));
    await fs.writeFile(result_watermark, await tensorToPNG(watermark));
});
