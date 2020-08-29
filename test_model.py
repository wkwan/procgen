import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

import numpy
import sys
numpy.set_printoptions(threshold=sys.maxsize)

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
#                'dog', 'frog', 'horse', 'ship', 'truck']

print(train_images.shape)




# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     # The CIFAR labels happen to be arrays, 
#     # which is why you need the extra index
#     plt.xlabel(class_names[train_labels[i][0]])
# plt.show()

# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# model.add(layers.MaxPooling2D((2, 2)))

# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10))

# model.summary()

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])




# history = model.fit(train_images, train_labels, epochs=1, 
#                     validation_data=(test_images, test_labels))

# get_3rd_layer_output = tf.keras.backend.function([model.layers[0].input],
#                                   [model.layers[3].output])
# layer_output = get_3rd_layer_output([train_images, train_labels])[0]
# print("3rd layer output")
# layer_output_np = tf.keras.backend.eval(layer_output)
# print(layer_output_np)

# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')

# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

# print(test_acc)





def conv_layer(depth, name):
    layer = tf.keras.layers.Conv2D( 
        filters=depth, kernel_size=3, strides=1, padding="same", name=name)
    return layer

def residual_block(x, depth, prefix):
    inputs = x
    assert inputs.get_shape()[-1].value == depth
    x = tf.keras.layers.ReLU()(x)
    x = conv_layer(depth, name=prefix + "_conv0")(x)
    x = tf.keras.layers.ReLU()(x)
    x = conv_layer(depth, name=prefix + "_conv1")(x)
    return x + inputs


def conv_sequence(x, depth, prefix):
    # print("x shape", x.shape)
    x = conv_layer(depth, prefix + "_conv")(x)
    # if "seq0" in prefix:
    #     # print("do seq 0")
    #     x = tf.reshape(x, tf.stack([tf.shape(x)[0], 128, 64, 3]))
    #     # print("reshaped", x.shape)
    #     x = conv_layer(depth, prefix + "_conv")(x)
    # else:
    #     # print("do ", prefix)
    #     x = conv_layer(depth, prefix + "_conv")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)
    x = residual_block(x, depth, prefix=prefix + "_block0")
    x = residual_block(x, depth, prefix=prefix + "_block1")
    return x


class ImpalaCNN():
    """
    Network from IMPALA paper implemented in ModelV2 API.

    Based on https://github.com/ray-project/ray/blob/master/rllib/models/tf/visionnet_v2.py
    and https://github.com/openai/baselines/blob/9ee399f5b20cd70ac0a871927a6cf043b478193f/baselines/common/models.py#L28
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        # print("INIT THE MODEL HEYYYYYYYYY")
        depths = [16, 32, 32]
        inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        scaled_inputs = tf.cast(inputs, tf.float32) / 255.0

        x = scaled_inputs

        for i, depth in enumerate(depths):
            x = conv_sequence(x, depth, prefix=f"seq{i}")

        x = tf.keras.layers.Flatten()(x)
        
        x = tf.keras.layers.ReLU()(x)
        

        #intermediate output:
        # self.base_model = tf.keras.Model(inputs=inputs, outputs=final_relu_layer.output)
        
        x = tf.keras.layers.Dense(units=256, activation="relu", name="hidden")(x)
        logits = tf.keras.layers.Dense(units=num_outputs, name="pi")(x)
        value = tf.keras.layers.Dense(units=1, name="vf")(x)
        self.base_model = tf.keras.Model(inputs, [logits, value])

        # for layer in self.base_model.layers:
        #     print("output of layer ", layer.output)
        # print(self.base_model.summary())
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        # explicit cast to float32 needed in eager
        obs = tf.cast(input_dict["obs"], tf.float32)
        # obs = tf.keras.backend.print_tensor(obs, message='obs')

        # intermediate_out = self.base_model.get_layer("seq1_block1_conv0").output
        # intermediate_model = self.base_model(obs, intermediate_out)
        # tf.keras.backend.print_tensor(intermediate_model, message="seq1 block 1 conv 0")

        logits, self._value = self.base_model(obs)
        # print(logits, self._value)
        return logits, state

    def value_function(self):
        return tf.reshape(self._value, [-1])


