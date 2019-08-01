import * as fs from 'fs-extra';
import * as path from 'path';
import * as _ from 'lodash';
import * as tf from '@tensorflow/tfjs-node-gpu';
import { DataProvider } from "./DataProvider";

/**
 * 判断传入的图片是否有水印
 * 
 * 10：acc=0.970 loss=0.0347 val_acc=0.960 val_loss=0.0399 
 * 20：acc=0.992 loss=0.0121 val_acc=0.984 val_loss=0.0167 
 */

const stackSize = 20;
const minTransparency = 0.4;
const allowDuplicate = true
const noWatermarkPercentage = 0.7;  //学习数据中包含的无水印图片
const trainingDataNumber = 10000;   //训练数据数量
const validationPercentage = 0.2;   //分割多少的训练数据出来用作验证
const tensorBoardPath = path.join(__dirname, '../../bin/training_result/tensorBoard/PredictHasWatermark');
const savedModelPath = path.join(__dirname, '../../bin/training_result/model/PredictHasWatermark');

/**
 * 训练模型
 */
async function training() {
    await fs.emptyDir(tensorBoardPath);
    await fs.emptyDir(savedModelPath);

    const dataProvider = new DataProvider(1, stackSize, minTransparency, allowDuplicate);

    const model = tf.sequential({ name: 'predict-has-watermark' });
    model.add(tf.layers.inputLayer({ inputShape: [3 * stackSize] }));
    model.add(tf.layers.dense({ units: 3 * stackSize, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'relu' }));
    model.compile({ optimizer: tf.train.adam(), loss: 'meanSquaredError', metrics: ['accuracy'] });
    model.summary();

    //生成训练数据
    const inputs: tf.Tensor1D[] = [], outputs: tf.Tensor1D[] = [];

    const hasWatermark = await dataProvider.getWaterMarkData(Math.round(trainingDataNumber * (1 - noWatermarkPercentage)));
    const noWatermark = await dataProvider.getNoWatermarkData(Math.round(trainingDataNumber * noWatermarkPercentage));
    hasWatermark.forEach(item => { item.answer = tf.tensor1d([1]) as any; });
    noWatermark.forEach(item => { item.answer = tf.tensor1d([0]) as any; });

    _.shuffle(_.concat(hasWatermark, noWatermark)).forEach(data => {
        inputs.push(tf.tidy(() => data.test.flatten()));
        outputs.push(data.answer as any);
        data.dispose();
    });
    dataProvider.dispose();

    const split = Math.floor(trainingDataNumber * (1 - validationPercentage));

    await model.fit(tf.stack(inputs.slice(0, split)), tf.stack(outputs.slice(0, split)), {
        epochs: 30,
        shuffle: true,
        validationData: [tf.stack(inputs.slice(split)), tf.stack(outputs.slice(split))],
        callbacks: tf.node.tensorBoard(tensorBoardPath)
    });

    await model.save('file://' + savedModelPath);
    console.log('训练完成');
}

training();