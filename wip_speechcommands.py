#%%
from IPython.display import Audio
from torchaudio.datasets import SPEECHCOMMANDS

dataset = SPEECHCOMMANDS(root="../data", download=True)
waveform, sr, *_ = dataset[0]
audio = Audio(waveform[0].numpy(), rate=sr)
# %%
labels = {
    "backward": 0,
    "bed": 1,
    "bird": 2,
    "cat": 3,
    "dog": 4,
    "down": 5,
    "eight": 6,
    "five": 7,
    "follow": 8,
    "forward": 9,
    "four": 10,
    "go": 11,
    "happy": 12,
    "house": 13,
    "learn": 14,
    "left": 15,
    "marvin": 16,
    "nine": 17,
    "no": 18,
    "off": 19,
    "on": 20,
    "one": 21,
    "right": 22,
    "seven": 23,
    "sheila": 24,
    "six": 25,
    "stop": 26,
    "three": 27,
    "tree": 28,
    "two": 29,
    "up": 30,
    "visual": 31,
    "wow": 32,
    "yes": 33,
    "zero": 34,
}
#%%
from torchaudio.transforms import MelSpectrogram

inp = MelSpectrogram(sample_rate=sr)(waveform)
from torch import nn

# %%
from torchvision.models import vgg16_bn

model = vgg16_bn(num_classes=35, pretrained=False)
model.features[0] = nn.Conv2d(
    in_channels=1,
    out_channels=model.features[0].out_channels,
    kernel_size=model.features[0].kernel_size,
    stride=model.features[0].stride,
    padding=model.features[0].padding)
# %%
