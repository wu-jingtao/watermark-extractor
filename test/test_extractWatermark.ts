import '@tensorflow/tfjs-node-gpu';
import * as path from 'path';
import * as fs from 'fs-extra';
import { findWatermark, extractWatermark, transformData } from '../src/Interface';

const _testingDataDir = path.join(__dirname, '../testing_data');
const _pictures = fs.readdirSync(_testingDataDir).map(item => path.join(_testingDataDir, item));

async function test_extractWatermark() {
    const savePath = path.join(__dirname, '../bin/testing_result/extractWatermark.png');
    await fs.ensureFile(savePath);
    await fs.remove(savePath);

    const data = await transformData(_pictures);
    const watermarkPosition = await findWatermark(data);
    await fs.writeFile(savePath, await extractWatermark(data, watermarkPosition));
    console.log('创建完成：extractWatermark.png');
}

test_extractWatermark().catch(console.error);