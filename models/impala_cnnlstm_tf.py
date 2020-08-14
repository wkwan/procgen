from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.models import ModelCatalog
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch
from baselines.a2c import utils


import numpy as np
tf = try_import_tf()

def conv_layer(depth, name):
    return tf.keras.layers.Conv2D( 
        filters=depth, kernel_size=3, strides=1, padding="same", name=name)

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


# class ImpalaCNN(TFModelV2):
#     """
#     Network from IMPALA paper implemented in ModelV2 API.

#     Based on https://github.com/ray-project/ray/blob/master/rllib/models/tf/visionnet_v2.py
#     and https://github.com/openai/baselines/blob/9ee399f5b20cd70ac0a871927a6cf043b478193f/baselines/common/models.py#L28
#     """

#     def __init__(self, obs_space, action_space, num_outputs, model_config, name):
#         super().__init__(obs_space, action_space, num_outputs, model_config, name)

#         depths = [16, 32, 32]
#         inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
#         scaled_inputs = tf.cast(inputs, tf.float32) / 255.0

#         x = scaled_inputs
#         for i, depth in enumerate(depths):
#             x = conv_sequence(x, depth, prefix=f"seq{i}")

#         x = tf.keras.layers.Flatten()(x)
#         x = tf.keras.layers.ReLU()(x)
#         x = tf.keras.layers.Dense(units=256, activation="relu", name="hidden")(x)
#         logits = tf.keras.layers.Dense(units=num_outputs, name="pi")(x)
#         value = tf.keras.layers.Dense(units=1, name="vf")(x)
#         self.base_model = tf.keras.Model(inputs, [logits, value])
#         self.register_variables(self.base_model.variables)

#     def forward(self, input_dict, state, seq_lens):
#         # explicit cast to float32 needed in eager
#         obs = tf.cast(input_dict["obs"], tf.float32)
#         logits, self._value = self.base_model(obs)
#         return logits, state

#     def value_function(self):
#         return tf.reshape(self._value, [-1])

# def cnn_lstm(nlstm=128, layer_norm=False, conv_fn=nature_cnn, **conv_kwargs):
#     def network_fn(X, nenv=1):
#         nbatch = X.shape[0]
#         nsteps = nbatch // nenv

#         h = conv_fn(X, **conv_kwargs)

#         M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
#         S = tf.placeholder(tf.float32, [nenv, 2*nlstm]) #states

#         xs = batch_to_seq(h, nenv, nsteps)
#         ms = batch_to_seq(M, nenv, nsteps)

#         if layer_norm:
#             h5, snew = utils.lnlstm(xs, ms, S, scope='lnlstm', nh=nlstm)
#         else:
#             h5, snew = utils.lstm(xs, ms, S, scope='lstm', nh=nlstm)

#         h = seq_to_batch(h5)
#         initial_state = np.zeros(S.shape.as_list(), dtype=float)

#         return h, {'S':S, 'M':M, 'state':snew, 'initial_state':initial_state}

#     return network_fn

# Register model in ModelCatalog

class ImpalaCNNLSTM(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)


        depths = [16, 32, 32]
        inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        scaled_inputs = tf.cast(inputs, tf.float32) / 255.0


        nbatch = inputs.shape[0]
        nenv = 1
        nsteps = nbatch // nenv
        nlstm = 256


        # original impala
        x = scaled_inputs
        for i, depth in enumerate(depths):
            x = conv_sequence(x, depth, prefix=f"seq{i}")

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dense(units=256, activation="relu", name="hidden")(x)

        #ending

        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [1, 2*nlstm]) #states

        xs = batch_to_seq(x, nenv, nsteps)
        ms = batch_to_seq(M, nenv, nsteps)

        h5, snew = utils.lstm(xs, ms, S, scope='lstm', nh=nlstm)

        h = seq_to_batch(h5)
        initial_state = np.zeros(S.shape.as_list(), dtype=float)

        return h, {'S':S, 'M':M, 'state':snew, 'initial_state':initial_state}

        logits = tf.keras.layers.Dense(units=num_outputs, name="pi")(h)
        value = tf.keras.layers.Dense(units=1, name="vf")(h)
        self.base_model = tf.keras.Model(inputs, [logits, value])
        self.register_variables(self.base_model.variables)


    def forward(self, input_dict, state, seq_lens):
        # explicit cast to float32 needed in eager
        obs = tf.cast(input_dict["obs"], tf.float32)
        logits, self._value = self.base_model(obs)
        return logits, state

    def value_function(self):
        return tf.reshape(self._value, [-1])

ModelCatalog.register_custom_model("impala_cnnlstm_tf", ImpalaCNNLSTM)
