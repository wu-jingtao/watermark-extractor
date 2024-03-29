import * as fs from 'fs-extra';
import * as path from 'path';
import * as _ from 'lodash';
import * as tf from '@tensorflow/tfjs-node';
import { DataProvider } from "./DataProvider";

/**
 * 5c：acc=0.749 loss=4.77e-3 val_acc=0.752 val_loss=4.75e-3
 * 5w：acc=1.00 loss=2.39e-4 val_acc=1.00 val_loss=2.46e-4 
 * 5b：acc=1.00 loss=2.77e-4 val_acc=1.00 val_loss=2.89e-4 
 */

const stackSize = 5;
const minTransparency = 0.5;
const trainingDataNumber = 20000;   //训练数据数量
const validationPercentage = 0.2;   //分割多少的训练数据出来用作验证
const tensorBoardDir = path.join(__dirname, '../../bin/training_result/tensorBoard/ExtractWatermark');
const savedModelDir = path.join(__dirname, '../../bin/training_result/model/ExtractWatermark');

async function training_extract_watermark(mode: 'colorful' | 'white' | 'black') {
    console.log('开始训练', mode);

    const tensorBoardPath = path.join(tensorBoardDir, mode);
    const savedModelPath = path.join(savedModelDir, mode);
    await fs.emptyDir(tensorBoardPath);
    await fs.emptyDir(savedModelPath);

    const dataProvider = new DataProvider(stackSize, minTransparency);

    const model = tf.sequential({ name: `extract-${mode}-watermark` });
    model.add(tf.layers.inputLayer({ inputShape: [3 * stackSize] }));
    model.add(tf.layers.dense({ units: 3 * stackSize, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 4 }));
    model.add(tf.layers.reLU({ maxValue: 1 }));
    model.compile({ optimizer: tf.train.adam(), loss: 'meanSquaredError', metrics: ['accuracy'] });
    model.summary();

    //生成训练数据
    const inputs: tf.Tensor1D[] = [], outputs: tf.Tensor1D[] = [];
    tf.tidy(() => {
        for (let i = 0; i < trainingDataNumber; i++) {
            const watermark = dataProvider[mode === 'colorful' ? 'colorPixel' : mode === 'white' ? 'whitePixel' : 'blackPixel'];
            const data = dataProvider.mixer(dataProvider.colorPixelStack, watermark);
            inputs.push(tf.keep(data.mixed.flatten()));
            outputs.push(tf.keep(data.watermark));
        }
    });

    const split = Math.floor(trainingDataNumber * (1 - validationPercentage));
    await model.fit(tf.stack(inputs.slice(0, split)), tf.stack(outputs.slice(0, split)), {
        epochs: 30,
        shuffle: true,
        validationData: [tf.stack(inputs.slice(split)), tf.stack(outputs.slice(split))],
        callbacks: tf.node.tensorBoard(tensorBoardPath)
    });

    await model.save('file://' + savedModelPath);
    console.log('训练完成', mode, '\n\n\n');
}

(async () => {
    try {
        await training_extract_watermark('colorful');
        await training_extract_watermark('white');
        await training_extract_watermark('black');
    } catch (error) {
        console.error(error);
    }
})();