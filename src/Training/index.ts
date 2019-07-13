import * as fs from 'fs-extra';
import * as path from 'path';
import * as _ from 'lodash';
import * as canvas from 'canvas';
import * as tf from '@tensorflow/tfjs-node';
import { DataProvider } from "./DataProvider";

const windowSize = 1;
const stackSize = 40;
const minTransparency = 0.4;
const noWatermarkPercentage = 0.3;  //学习数据中包含的无水印图片
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

    const inputPixelNumber = windowSize ** 2 * 3 * stackSize;
    const outputPixelNumber = windowSize ** 2 * 4;

    const dataProvider = new DataProvider(windowSize, stackSize, minTransparency);

    const model = tf.sequential({ name: 'watermark-extractor' });
    model.add(tf.layers.inputLayer({ inputShape: [inputPixelNumber] }));
    model.add(tf.layers.dense({ units: inputPixelNumber, activation: 'relu' }));
    model.add(tf.layers.dense({ units: outputPixelNumber, useBias: true }));
    model.add(tf.layers.reLU({ maxValue: 1 }));
    model.add(tf.layers.reshape({ targetShape: [windowSize, windowSize, 4] }));
    model.compile({ optimizer: tf.train.adam(), loss: 'meanSquaredError', metrics: ['accuracy'] });
    model.summary();

    //生成训练数据
    const inputs = [], outputs = [];
    for (let i = 0, j = Math.round(trainingDataNumber * (1 - noWatermarkPercentage)); i < j; i++) {
        const data = await dataProvider.getWaterMarkData();
        inputs.push(tf.tidy(() => data.test.flatten()));
        outputs.push(data.answer);
    }
    for (let i = 0, j = Math.round(trainingDataNumber * noWatermarkPercentage); i < j; i++) {
        const data = await dataProvider.getNoWatermarkData();
        inputs.push(tf.tidy(() => data.test.flatten()));
        outputs.push(data.answer);
    }

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

    const pieceNumber = Math.trunc(150 / windowSize);
    const pictureSize = windowSize * pieceNumber;
    const dataProvider = new DataProvider(pictureSize, stackSize, minTransparency);
    const model = await tf.loadLayersModel('file://' + path.join(savedModelPath, 'model.json'));
    const can = canvas.createCanvas(pictureSize, pictureSize);
    const ctx = can.getContext('2d');

    const data = _.concat(await Promise.all(_.times(10, () => dataProvider.getWaterMarkData())), await Promise.all(_.times(3, () => dataProvider.getNoWatermarkData())));

    for (let i = 0; i < data.length; i++) {
        const result = tf.tidy(() => {
            const result_row: tf.Tensor4D[] = [];
            for (const row of data[i].test.split(pieceNumber)) {
                const result_cell: tf.Tensor4D[] = [];
                for (const cell of row.split(pieceNumber, 1)) {
                    result_cell.push(model.predict(cell.flatten().expandDims()) as any);
                }
                result_row.push(tf.concat(result_cell as any, 2));
            }

            return tf.concat(result_row, 1).squeeze([0]);
        });

        const real = tf.tidy(() => data[i].answer.mul(255).floor().cast('int32')) as tf.Tensor3D;
        const predict = tf.tidy(() => result.mul(255).floor().cast('int32')) as tf.Tensor3D;

        ctx.clearRect(0, 0, pictureSize, pictureSize);
        ctx.putImageData(canvas.createImageData(await tf.browser.toPixels(real), pictureSize, pictureSize), 0, 0);
        await fs.writeFile(path.join(checkPath, `${i}-r.png`), can.toBuffer('image/png'));

        ctx.clearRect(0, 0, pictureSize, pictureSize);
        ctx.putImageData(canvas.createImageData(await tf.browser.toPixels(predict), pictureSize, pictureSize), 0, 0);
        await fs.writeFile(path.join(checkPath, `${i}-p.png`), can.toBuffer('image/png'));

        real.dispose();
        predict.dispose();

        console.log('完成：', i + 1);
    }

    console.log('完成');
}

training();
//check();