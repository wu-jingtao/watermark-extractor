import * as fs from 'fs-extra';
import * as path from 'path';
import * as canvas from 'canvas';
import * as tf from '@tensorflow/tfjs-node';
import log from 'log-formatter';
import { DataProvider } from "./DataProvider";

/**
 * 训练模型
 */
async function training() {
    const windowSize = 1;
    const stackSize = 5;
    const noWatermarkPercentage = 0.3;  //学习数据中包含的无水印图片
    const trainingDataNumber = 10000;   //训练数据数量
    const validationPercentage = 0.2;   //分割多少的训练数据出来用作验证
    const tensorBoardPath = path.join(__dirname, '../../bin/training_result/tensorBoard');
    const savedModelPath = path.join(__dirname, '../../bin/training_result/model');

    await fs.emptyDir(tensorBoardPath);
    await fs.emptyDir(savedModelPath);

    const dataProvider = new DataProvider(windowSize, stackSize);

    const model = tf.sequential({ name: 'watermark-extractor' });
    model.add(tf.layers.flatten({ inputShape: [windowSize, windowSize, 3, stackSize] }));
    model.add(tf.layers.dense({ units: windowSize ** 2 * 3 * stackSize, activation: 'relu' }));
    model.add(tf.layers.dense({ units: windowSize ** 2 * 4, activation: 'relu' }));
    model.add(tf.layers.reshape({ targetShape: [windowSize, windowSize, 4] }));
    model.compile({ optimizer: tf.train.adam(), loss: 'meanSquaredError', metrics: ['accuracy'] });
    model.summary();

    //生成训练数据
    const inputs = [], outputs = [];
    for (let i = 0, j = Math.round(trainingDataNumber * (1 - noWatermarkPercentage)); i < j; i++) {
        const data = await dataProvider.getWaterMarkData();
        inputs.push(data.test);
        outputs.push(data.answer);
    }
    for (let i = 0, j = Math.round(trainingDataNumber * noWatermarkPercentage); i < j; i++) {
        const data = await dataProvider.getNoWatermarkData();
        inputs.push(data.test);
        outputs.push(data.answer);
    }

    const split = Math.floor(trainingDataNumber * (1 - validationPercentage));
    await model.fit(tf.stack(inputs.slice(0, split)), tf.stack(outputs.slice(0, split)), {
        epochs: 100,
        shuffle: true,
        validationData: [tf.stack(inputs.slice(split, trainingDataNumber)), tf.stack(outputs.slice(split, trainingDataNumber))],
        callbacks: tf.node.tensorBoard(tensorBoardPath)
    });

    await model.save('file://' + savedModelPath);
    log.green.bold('训练完成');
}

/**
 * 检测模型效果
 */
async function check() {
    const windowSize = 1;
    const stackSize = 5;
}

training();