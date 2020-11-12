import tensorflow as tf
from time import perf_counter

# with tf.device("/cpu:0"):
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation="relu"),
  tf.keras.layers.Dropout(0.2), # randomly deactivates 20% of neurons to prevent overfitting
  tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

# --------- TRAINING ---------

NUM_EPOCHS = 10

trainStartTime = perf_counter()
model.fit(x_train, y_train, epochs=10) # x_train is inputs, y_train is targets ("correct answers")
trainEndTime = perf_counter()
trainDurationMs = (trainEndTime - trainStartTime) * 1e3
print(f"Total training time for {NUM_EPOCHS} epochs: {trainDurationMs} ms")
print(f"Time per epoch: {trainDurationMs / NUM_EPOCHS} ms")
print(f"Training time per image: {trainDurationMs / x_train.shape[0]} ms")

# --------- EVALUATION ---------

evalStartTime = perf_counter()
evalOutput = model.evaluate(x_test, y_test)
evalEndTime = perf_counter()
evalDurationMs = (evalEndTime - evalStartTime) * 1e3
print(f"Total evaluation time: {evalDurationMs} ms")
print(f"Evaluation time per image: {evalDurationMs / x_test.shape[0]} ms")

print("\nEvaluation result:\n" +
  f"Loss = {round(evalOutput[0], 3)}; Accuracy = {round(evalOutput[1], 3)}")
