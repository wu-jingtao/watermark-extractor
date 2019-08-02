import * as fs from 'fs-extra';
import * as path from 'path';
import * as _ from 'lodash';
import * as tf from '@tensorflow/tfjs-node-gpu';
import { DataProvider } from "./DataProvider";

/**
 * 15： acc=0.778 loss=0.0991 val_acc=0.777 val_loss=0.0994
 * 20： acc=0.817 loss=4.94e-3 val_acc=0.783 val_loss=5.24e-3 
 * 20双层： acc=0.861 loss=2.35e-3 val_acc=0.814 val_loss=3.68e-3 出现过度拟合了
 * 20单层2倍： acc=0.835 loss=3.47e-3 val_acc=0.809 val_loss=4.11e-3 出现过度拟合了
 * 30： acc=0.827 loss=4.23e-3 val_acc=0.762 val_loss=5.53e-3
 * 40： acc=0.825 loss=3.41e-3 val_acc=0.830 val_loss=4.28e-3 
 * 70： acc=0.776 loss=0.0993 val_acc=0.762 val_loss=0.0984
 */

const stackSize = 20;
const minTransparency = 0.4;
const allowDuplicate = true;
const trainingDataNumber = 10000;   //训练数据数量
const validationPercentage = 0.2;   //分割多少的训练数据出来用作验证
const tensorBoardPath = path.join(__dirname, '../../bin/training_result/tensorBoard/ExtractWatermark');
const savedModelPath = path.join(__dirname, '../../bin/training_result/model/ExtractWatermark');

async function training() {
    console.log('开始训练');
    await fs.emptyDir(tensorBoardPath);
    await fs.emptyDir(savedModelPath);

    const dataProvider = new DataProvider(stackSize, minTransparency, allowDuplicate);

    const model = tf.sequential({ name: 'extract-watermark' });
    model.add(tf.layers.inputLayer({ inputShape: [3 * stackSize] }));
    model.add(tf.layers.dense({ units: 3 * stackSize, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 4 }));
    model.add(tf.layers.reLU({ maxValue: 1 }));
    model.compile({ optimizer: tf.train.adam(), loss: 'meanSquaredError', metrics: ['accuracy'] });
    model.summary();

    //生成训练数据
    const inputs = [], outputs = [];
    for (const item of await dataProvider.getWaterMarkData(trainingDataNumber)) {
        inputs.push(item.test.flatten());
        outputs.push(item.answer);
        item.test.dispose();
    }
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

training().catch(console.error);