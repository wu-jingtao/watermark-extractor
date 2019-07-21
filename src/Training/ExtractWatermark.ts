import * as fs from 'fs-extra';
import * as path from 'path';
import * as _ from 'lodash';
import * as canvas from 'canvas';
import * as tf from '@tensorflow/tfjs-node-gpu';
import { DataProvider } from "./DataProvider";

const stackSize = 5;
const minTransparency = 0.4;
const trainingDataNumber = 10000;   //训练数据数量
const validationPercentage = 0.2;   //分割多少的训练数据出来用作验证
const tensorBoardPath = path.join(__dirname, '../../bin/training_result/tensorBoard');
const savedModelPath = path.join(__dirname, '../../bin/training_result/model');
const checkPath = path.join(__dirname, '../../bin/training_result/checkPath');

/**
 * 训练模型
 */
async function training() {
    await fs.emptyDir(tensorBoardPath);
    await fs.emptyDir(savedModelPath);

    const dataProvider = new DataProvider(1, stackSize, minTransparency);

    const model = tf.sequential({ name: 'extract-watermark' });
    model.add(tf.layers.inputLayer({ inputShape: [3 * stackSize] }));
    model.add(tf.layers.dense({ units: 3 * stackSize, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 4, useBias: true }));
    model.add(tf.layers.reLU({ maxValue: 1 }));
    model.compile({ optimizer: tf.train.adam(), loss: 'meanSquaredError', metrics: ['accuracy'] });
    model.summary();

    //生成训练数据
    const inputs = [], outputs = [];
    for (const item of await dataProvider.getWaterMarkData(trainingDataNumber)) {
        inputs.push(item.test.flatten());
        outputs.push(item.answer.flatten());
        item.dispose();
    }
    dataProvider.dispose();

    const split = Math.floor(trainingDataNumber * (1 - validationPercentage));
    await model.fit(tf.stack(inputs.slice(0, split)), tf.stack(outputs.slice(0, split)), {
        epochs: 30,
        shuffle: true,
        validationData: [tf.stack(inputs.slice(split, trainingDataNumber)), tf.stack(outputs.slice(split, trainingDataNumber))],
        callbacks: tf.node.tensorBoard(tensorBoardPath)
    });

    await model.save('file://' + savedModelPath);
    console.log('训练完成');
}

/**
 * 检测模型效果
 */
async function check() {
    await fs.emptyDir(checkPath);

    const pictureSize = 150;

    const dataProvider = new DataProvider(pictureSize, stackSize, minTransparency);
    const model = await tf.loadLayersModel('file://' + path.join(savedModelPath, 'model.json'));
    const can = canvas.createCanvas(pictureSize, pictureSize);
    const ctx = can.getContext('2d');

    const data = await dataProvider.getWaterMarkData(10);
    dataProvider.dispose();

    for (let i = 0; i < data.length; i++) {
        const result = tf.tidy(() => {
            const input = data[i].test.reshape([pictureSize ** 2, 3 * stackSize]);
            const output = model.predict(input) as tf.Tensor2D;
            return output.reshape([pictureSize, pictureSize, 4]);
        });

        const real = tf.tidy(() => data[i].answer.mul(255).floor().cast('int32')) as tf.Tensor3D;
        const predict = tf.tidy(() => result.mul(255).floor().cast('int32')) as tf.Tensor3D;

        ctx.clearRect(0, 0, pictureSize, pictureSize);
        ctx.putImageData(canvas.createImageData(await tf.browser.toPixels(real), pictureSize, pictureSize), 0, 0);
        await fs.writeFile(path.join(checkPath, `${i}-r.png`), can.toBuffer('image/png'));

        ctx.clearRect(0, 0, pictureSize, pictureSize);
        ctx.putImageData(canvas.createImageData(await tf.browser.toPixels(predict), pictureSize, pictureSize), 0, 0);
        await fs.writeFile(path.join(checkPath, `${i}-p.png`), can.toBuffer('image/png'));

        result.dispose();
        real.dispose();
        predict.dispose();

        console.log('完成：', i + 1);
    }

    console.log('完成');
}

//training();
check();