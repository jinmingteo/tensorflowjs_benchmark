const tf = require("@tensorflow/tfjs-node");

const model = tf.sequential({
  layers: [
    tf.layers.flatten({ inputShape: [28, 28, 1] }),
    tf.layers.dense({ units: 128, activation: "relu" }),
    tf.layers.dropout({ rate: 0.2 }),
    tf.layers.dense({ units: 10, activation: "softmax" }),
  ],
});

model.compile({
  optimizer: "adam",
  loss: "categoricalCrossentropy",
  metrics: ["accuracy"],
});

module.exports = model;
