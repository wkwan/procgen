from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_torch

torch, nn = try_import_torch()
import math

class ResidualBlock(nn.Module):
    def __init__(self, channels, scale):
        super().__init__()
        print("residual block scale", scale)
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv0.weight.data *= scale / self.conv0.weight.norm(dim=(1, 2, 3), p=2, keepdim=True)
        self.conv0.bias.data *= 0

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1.weight.data *= scale / self.conv1.weight.norm(dim=(1, 2, 3), p=2, keepdim=True)
        self.conv1.bias.data *= 0
    
    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)
        scale = math.sqrt(1 / (math.sqrt(2) * math.sqrt(3)))
        print("first conv seq scale", scale)

        self.conv.weight.data *= scale / self.conv.weight.norm(dim=(1, 2, 3), p=2, keepdim=True)
        self.conv.bias.data *= 0
        scale = math.sqrt(scale)
        self.res_block0 = ResidualBlock(self._out_channels, scale)
        self.res_block1 = ResidualBlock(self._out_channels, scale)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


def parse_dtype(x):
    if isinstance(x, torch.dtype):
        return x
    elif isinstance(x, str):
        if x == "float32" or x == "float":
            return torch.float32
        elif x == "float64" or x == "double":
            return torch.float64
        elif x == "float16" or x == "half":
            return torch.float16
        elif x == "uint8":
            return torch.uint8
        elif x == "int8":
            return torch.int8
        elif x == "int16" or x == "short":
            return torch.int16
        elif x == "int32" or x == "int":
            return torch.int32
        elif x == "int64" or x == "long":
            return torch.int64
        elif x == "bool":
            return torch.bool
        else:
            raise ValueError(f"cannot parse {x} as a dtype")
    else:
        raise TypeError(f"cannot parse {type(x)} as dtype")

def NormedLinear(*args, scale=1.0, dtype=torch.float32, **kwargs):
    """
    nn.Linear but with normalized fan-in init
    """
    dtype = parse_dtype(dtype)
    if dtype == torch.float32:
        out = nn.Linear(*args, **kwargs)
    elif dtype == torch.float16:
        out = LinearF16(*args, **kwargs)
    else:
        raise ValueError(dtype)
    out.weight.data *= scale / out.weight.norm(dim=1, p=2, keepdim=True)
    if kwargs.get("bias", True):
        out.bias.data *= 0
    return out

class ImpalaCNN(TorchModelV2, nn.Module):
    """
    Network from IMPALA paper implemented in ModelV2 API.

    Based on https://github.com/ray-project/ray/blob/master/rllib/models/tf/visionnet_v2.py
    and https://github.com/openai/baselines/blob/9ee399f5b20cd70ac0a871927a6cf043b478193f/baselines/common/models.py#L28
    """



    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):

        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        h, w, c = obs_space.shape
        shape = (c, h, w)

        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        self.conv_seqs = nn.ModuleList(conv_seqs)
        self.hidden_fc = nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256)
        self.hidden_fc.weight.data *= 1.4 / self.hidden_fc.weight.norm(dim=1, p=2, keepdim=True)
        self.hidden_fc.bias.data *= 0

        self.logits_fc = nn.Linear(in_features=256, out_features=num_outputs)
        self.logits_fc.weight.data *= 0.1 / self.logits_fc.weight.norm(dim=1, p=2, keepdim=True)
        self.logits_fc.bias.data * 0

        self.value_fc = nn.Linear(in_features=256, out_features=1)
        self.value_fc.weight.data *= 0.1 / self.value_fc.weight.norm(dim=1, p=2, keepdim=True)
        self.value_fc.bias.data *= 0

        self.aux_vf_head = NormedLinear(256, 1, scale=0.1)
        
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()
        # print("x shape", x.shape)
        x = x / 255.0  # scale to 0-1
        x = x.permute(0, 3, 1, 2)  # NHWC => NCHW
        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
        x = torch.flatten(x, start_dim=1)
        x = nn.functional.relu(x)
        x = self.hidden_fc(x)
        x = nn.functional.relu(x)
        logits = self.logits_fc(x)
        # print("x before detach", x)
        x.detach() #detach during policy phase only
        # print("x after detach", x)
        value = self.value_fc(x)
        # print("value before squeeze", value)
        self._value = value.squeeze(1)
        # print("value after squeeze", self._value)
        # print("logits", logits)
        return logits, state

    def forward_aux(self, input_dict):
        x = input_dict["obs"].float()
        # print("x shape", x.shape)
        x = x / 255.0  # scale to 0-1
        x = x.permute(0, 3, 1, 2)  # NHWC => NCHW
        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
        x = torch.flatten(x, start_dim=1)
        x = nn.functional.relu(x)
        x = self.hidden_fc(x)
        x = nn.functional.relu(x)
        logits = self.logits_fc(x)
        value = self.value_fc(x)
        self._value = value.squeeze(1)
        return logits, self.aux_vf_head(x).squeeze(1)

    @override(TorchModelV2)
    def value_function(self):
        assert self._value is not None, "must call forward() first"
        return self._value




ModelCatalog.register_custom_model("impala_cnn_torch", ImpalaCNN)