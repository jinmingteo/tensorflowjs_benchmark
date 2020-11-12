// const tf = require("@tensorflow/tfjs-node-gpu");
const process = require("process");

const data = require("./data");
const model = require("./model");

const NUM_EPOCHS = 10;

const run = async ({ epochs, batchSize = 32, modelSavePath = null }) => {
  model.summary();

  await data.loadData();

  /* --------- TRAINING --------- */

  const { images: trainImages, labels: trainLabels } = data.getTrainData();

  const trainStartTime = process.hrtime.bigint();
  await model.fit(trainImages, trainLabels, { epochs, batchSize });
  const trainEndTime = process.hrtime.bigint();
  const trainDurationMs = (trainEndTime - trainStartTime) / 1e6;
  console.log(`Total training time for ${NUM_EPOCHS} epochs: ${trainDurationMs} ms`);
  console.log(`Time per epoch: ${trainDurationMs / BigInt(NUM_EPOCHS)} ms`);
  console.log(`Training time per image: ${trainDurationMs / BigInt(trainImages.shape[0])} ms`);

  /* --------- EVALUATION --------- */

  const { images: testImages, labels: testLabels } = data.getTestData();

  const evalStartTime = process.hrtime.bigint();
  const evalOutput = model.evaluate(testImages, testLabels);
  const evalEndTime = process.hrtime.bigint();
  const evalDurationMs = (evalEndTime - evalStartTime) / 1e6;
  console.log(`Total evaluation time: ${evalDurationMs} ms`);
  console.log(`Evaluation time per image: ${evalDurationMs / BigInt(testImages.shape[0])} ms`);

  console.log("\nEvaluation result:\n" +
    `  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; `+
    `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`);
};

run({ epochs: NUM_EPOCHS });
