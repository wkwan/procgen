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
    print("x shape", x.shape)
    if "seq0" in prefix:
        print("do seq 0")
        x = tf.reshape(x, (1, 256, 64, 3))
        print("reshaped", x.shape)
        x = conv_layer(depth, prefix + "_conv")(x)
    else:
        print("do ", prefix)
        x = conv_layer(depth, prefix + "_conv")(x)
    print("conv layer shaper init", x.shape)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)
    x = residual_block(x, depth, prefix=prefix + "_block0")
    x = residual_block(x, depth, prefix=prefix + "_block1")
    print("conv layer shape after resid", x.shape)
    return x


class ImpalaCNN(TFModelV2):
    """
    Network from IMPALA paper implemented in ModelV2 API.

    Based on https://github.com/ray-project/ray/blob/master/rllib/models/tf/visionnet_v2.py
    and https://github.com/openai/baselines/blob/9ee399f5b20cd70ac0a871927a6cf043b478193f/baselines/common/models.py#L28
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        depths = [16, 32, 32]
        print("the shape is ", obs_space.shape)
        inputs = tf.keras.layers.Input(shape=(4, 64, 64, 3), name="observations")
        print("inputs shape", inputs.shape)

        scaled_inputs = tf.cast(inputs, tf.float32) / 255.0
        print("scaled inputs shape ", scaled_inputs.shape)

        # scaled_inputs = tf.reshape(scaled_inputs, (4, 64, 64, 3))

        x = scaled_inputs
        for i, depth in enumerate(depths):
            x = conv_sequence(x, depth, prefix=f"seq{i}")

        x = tf.keras.layers.Flatten()(x)
        print("flattened")
        x = tf.keras.layers.ReLU()(x)
        print("relued")
        x = tf.keras.layers.Dense(units=256, activation="relu", name="hidden")(x)
        print("densed")
        logits = tf.keras.layers.Dense(units=num_outputs, name="pi")(x)
        print("logited")
        value = tf.keras.layers.Dense(units=1, name="vf")(x)
        print("valued")
        self.base_model = tf.keras.Model(inputs, [logits, value])
        print("base modelled")
        self.register_variables(self.base_model.variables)
        print("registered")

    def forward(self, input_dict, state, seq_lens):
        # explicit cast to float32 needed in eager
        print("input dict shape", input_dict["obs"].shape)
        # input_dict["obs"] = tf.squeeze(input_dict["obs"], 1)
        # print("input keys", input_dict.keys())

        # print("input dict shape after squeeze", input_dict["obs"].shape)

        obs = tf.cast(input_dict["obs"], tf.float32)
        logits, self._value = self.base_model(obs)
        return logits, state

    def value_function(self):
        print("value function")
        return tf.reshape(self._value, [-1])


# Register model in ModelCatalog
ModelCatalog.register_custom_model("impala_cnn_tf", ImpalaCNN)
