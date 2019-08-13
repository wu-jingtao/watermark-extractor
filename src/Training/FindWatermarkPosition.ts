import * as fs from 'fs-extra';
import * as path from 'path';
import * as _ from 'lodash';
import * as tf from '@tensorflow/tfjs-node';
import { DataProvider } from "./DataProvider";

/**
 * 找图图片中水印所在的位置
 * 
 * 5c：acc=0.929 loss=0.0540 val_acc=0.925 val_loss=0.0578 
 * 5w：acc=0.999 loss=8.05e-4 val_acc=0.999 val_loss=1.09e-3  
 * 5b：acc=0.997 loss=2.74e-3 val_acc=0.993 val_loss=5.51e-3
 */

const stackSize = 5;
const minTransparency = 0.4;
const noWatermarkPercentage = 0.5;  //学习数据中包含的无水印图片
const trainingDataNumber = 20000;   //训练数据数量
const validationPercentage = 0.2;   //分割多少的训练数据出来用作验证
const tensorBoardDir = path.join(__dirname, '../../bin/training_result/tensorBoard/FindWatermarkPosition');
const savedModelDir = path.join(__dirname, '../../bin/training_result/model/FindWatermarkPosition');

/**
 * 训练识别水印
 */
async function training_find_watermark_position(mode: 'colorful' | 'white' | 'black') {
    console.log('开始训练', mode);

    const tensorBoardPath = path.join(tensorBoardDir, mode);
    const savedModelPath = path.join(savedModelDir, mode);
    await fs.emptyDir(tensorBoardPath);
    await fs.emptyDir(savedModelPath);

    const dataProvider = new DataProvider(stackSize, minTransparency);

    const model = tf.sequential({ name: `find-${mode}-watermark` });
    model.add(tf.layers.inputLayer({ inputShape: [3 * stackSize] }));
    model.add(tf.layers.dense({ units: 3 * stackSize, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1 }));
    model.add(tf.layers.reLU({ maxValue: 1 }));
    model.compile({ optimizer: tf.train.adam(), loss: 'meanSquaredError', metrics: ['accuracy'] });
    model.summary();

    //生成训练数据
    const trainingData: { test: tf.Tensor1D, answer: tf.Tensor1D }[] = [];
    tf.tidy(() => {
        for (let i = 0; i < Math.round(trainingDataNumber * noWatermarkPercentage); i++) {
            const watermark = dataProvider[mode === 'colorful' ? 'colorPixel' : mode === 'white' ? 'whitePixel' : 'blackPixel'];
            const data = dataProvider.mixer(dataProvider.colorPixelStack, watermark);
            trainingData.push({ test: tf.keep(data.mixed.flatten()), answer: tf.keep(tf.tensor1d([1])) });
        }

        for (let i = 0; i < Math.round(trainingDataNumber * (1 - noWatermarkPercentage)); i++) {
            trainingData.push({ test: tf.keep(dataProvider.colorPixelStack.flatten()), answer: tf.keep(tf.tensor1d([0])) });
        }
    });

    const inputs: tf.Tensor1D[] = [], outputs: tf.Tensor1D[] = [];
    _.shuffle(trainingData).forEach(data => { inputs.push(data.test); outputs.push(data.answer); });

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
        await training_find_watermark_position('colorful');
        await training_find_watermark_position('white');
        await training_find_watermark_position('black');
    } catch (error) {
        console.error(error);
    }
})();
