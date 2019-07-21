import * as fs from 'fs-extra';
import * as path from 'path';
import * as _ from 'lodash';
import * as canvas from 'canvas';
import * as tf from '@tensorflow/tfjs-node-gpu';
import { DataProvider } from "./DataProvider";

/**
 * 提取图片中的水印
 * 
 * 把每个颜色单独提出来学习是一个坏主意，效果奇差无比，正确率只有0.1%。
 * 看来这项任务并不是简单的做做颜色加减就可以了的，其背后还有更深奥的原理
 */

const stackSize = 5;
const minTransparency = 0.4;
const trainingDataNumber = 10000;   //训练数据数量
const validationPercentage = 0.2;   //分割多少的训练数据出来用作验证
const tensorBoardPath = path.join(__dirname, '../../bin/training_result/tensorBoard/ExtractWatermark');
const savedModelPath = path.join(__dirname, '../../bin/training_result/model/ExtractWatermark');
const checkPath = path.join(__dirname, '../../bin/training_result/check/ExtractWatermark');

/**
 * 训练模型
 */
async function training() {
    await fs.emptyDir(tensorBoardPath);
    await fs.emptyDir(savedModelPath);

    const dataProvider = new DataProvider(1, stackSize, minTransparency);

    const input_r = tf.input({ shape: [stackSize] });
    const input_g = tf.input({ shape: [stackSize] });
    const input_b = tf.input({ shape: [stackSize] });
    const input_a = tf.input({ shape: [stackSize * 3] });

    const middle_r = tf.layers.dense({ units: stackSize * 3, activation: 'relu' }).apply(input_r);
    const middle_g = tf.layers.dense({ units: stackSize * 3, activation: 'relu' }).apply(input_g);
    const middle_b = tf.layers.dense({ units: stackSize * 3, activation: 'relu' }).apply(input_b);
    const middle_a = tf.layers.dense({ units: stackSize * 3 * 3, activation: 'relu' }).apply(input_a);

    const middle2_r = tf.layers.dense({ units: 1, useBias: true }).apply(middle_r);
    const middle2_g = tf.layers.dense({ units: 1, useBias: true }).apply(middle_g);
    const middle2_b = tf.layers.dense({ units: 1, useBias: true }).apply(middle_b);
    const middle2_a = tf.layers.dense({ units: 1, useBias: true }).apply(middle_a);

    const output_r = tf.layers.reLU({ maxValue: 1, name: 'r' }).apply(middle2_r);
    const output_g = tf.layers.reLU({ maxValue: 1, name: 'g' }).apply(middle2_g);
    const output_b = tf.layers.reLU({ maxValue: 1, name: 'b' }).apply(middle2_b);
    const output_a = tf.layers.reLU({ maxValue: 1, name: 'a' }).apply(middle2_a);

    const model = tf.model({
        name: 'extract-watermark',
        inputs: [input_r, input_g, input_b, input_a],
        outputs: [output_r, output_g, output_b, output_a] as any
    });

    model.compile({ optimizer: tf.train.adam(), loss: 'meanSquaredError', metrics: ['accuracy'] });
    model.summary();

    //生成训练数据
    const data = await dataProvider.getWaterMarkData(trainingDataNumber);
    dataProvider.dispose();

    const trainingData = tf.tidy(() => {
        const temp_data_x_r = [];
        const temp_data_x_g = [];
        const temp_data_x_b = [];
        const temp_data_x_a = [];
        const temp_data_y_r = [];
        const temp_data_y_g = [];
        const temp_data_y_b = [];
        const temp_data_y_a = [];

        for (const item of data) {
            const [xr, xg, xb] = item.test.split(3, 2);
            temp_data_x_r.push(xr.flatten());
            temp_data_x_g.push(xg.flatten());
            temp_data_x_b.push(xb.flatten());
            temp_data_x_a.push(item.test.flatten());

            const [yr, yg, yb, ya] = item.answer.split(4, 2);
            temp_data_y_r.push(yr.flatten());
            temp_data_y_g.push(yg.flatten());
            temp_data_y_b.push(yb.flatten());
            temp_data_y_a.push(ya.flatten());

            item.dispose();
        }

        const split = Math.floor(trainingDataNumber * (1 - validationPercentage));

        const data_r_x1 = tf.stack(temp_data_x_r.slice(0, split)), data_r_y1 = tf.stack(temp_data_y_r.slice(0, split));
        const data_g_x1 = tf.stack(temp_data_x_g.slice(0, split)), data_g_y1 = tf.stack(temp_data_y_g.slice(0, split));
        const data_b_x1 = tf.stack(temp_data_x_b.slice(0, split)), data_b_y1 = tf.stack(temp_data_y_b.slice(0, split));
        const data_a_x1 = tf.stack(temp_data_x_a.slice(0, split)), data_a_y1 = tf.stack(temp_data_y_a.slice(0, split));
        const data_r_x2 = tf.stack(temp_data_x_r.slice(split)), data_r_y2 = tf.stack(temp_data_y_r.slice(split));
        const data_g_x2 = tf.stack(temp_data_x_g.slice(split)), data_g_y2 = tf.stack(temp_data_y_g.slice(split));
        const data_b_x2 = tf.stack(temp_data_x_b.slice(split)), data_b_y2 = tf.stack(temp_data_y_b.slice(split));
        const data_a_x2 = tf.stack(temp_data_x_a.slice(split)), data_a_y2 = tf.stack(temp_data_y_a.slice(split));

        return {
            x1: [data_r_x1, data_g_x1, data_b_x1, data_a_x1],
            x2: [data_r_x2, data_g_x2, data_b_x2, data_a_x2],
            y1: [data_r_y1, data_g_y1, data_b_y1, data_a_y1],
            y2: [data_r_y2, data_g_y2, data_b_y2, data_a_y2],
        }
    });

    await model.fit(trainingData.x1, trainingData.y1, {
        epochs: 30,
        shuffle: true,
        validationData: [trainingData.x2, trainingData.y2],
        callbacks: tf.node.tensorBoard(tensorBoardPath)
    });

    await model.save('file://' + savedModelPath);
    console.log('训练完成');
}

training();