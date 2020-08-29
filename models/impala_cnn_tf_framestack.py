from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.models import ModelCatalog

tf = try_import_tf()

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
    if "seq0" in prefix:
        x = tf.reshape(x, tf.stack([tf.shape(x)[0], 128, 64, 3]))
        x = conv_layer(depth, prefix + "_conv")(x)
    else:
        x = conv_layer(depth, prefix + "_conv")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)
    x = residual_block(x, depth, prefix=prefix + "_block0")
    x = residual_block(x, depth, prefix=prefix + "_block1")
    return x


class ImpalaCNN(TFModelV2):
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

        # x = tf.keras.backend.print_tensor(x, message='last conv')

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


# Register model in ModelCatalog
ModelCatalog.register_custom_model("impala_cnn_tf_framestack", ImpalaCNN)