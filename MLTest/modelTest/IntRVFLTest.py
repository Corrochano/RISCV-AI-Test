import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import MNIST
# Note: this example requires the torchmetrics library: https://torchmetrics.readthedocs.io
import torchmetrics
from tqdm import tqdm
from torch.profiler import profile, ProfilerActivity, record_function

import torchhd
from torchhd.models import IntRVFL
from torchhd import embeddings

# Use the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

DIMENSIONS = 10000
IMG_SIZE = 28
NUM_LEVELS = 1000
BATCH_SIZE = 1  # for GPUs with enough memory we can process multiple images at ones

transform = torchvision.transforms.ToTensor()

train_ds = MNIST("../data", train=True, transform=transform, download=True)
train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

test_ds = MNIST("../data", train=False, transform=transform, download=True)
test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

class Encoder(nn.Module):
    def __init__(self, out_features, size, levels):
        super(Encoder, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.position = embeddings.Random(size * size, out_features)
        self.value = embeddings.Level(levels, out_features)

    def forward(self, x):
        x = self.flatten(x)
        sample_hv = torchhd.bind(self.position.weight, self.value(x))
        sample_hv = torchhd.multiset(sample_hv)
        return torchhd.hard_quantize(sample_hv)

encode = Encoder(DIMENSIONS, IMG_SIZE, NUM_LEVELS)
encode = encode.to(device)

num_classes = len(train_ds.classes)
model = IntRVFL(DIMENSIONS, DIMENSIONS, num_classes) # in_features, dimensions, out_features
model = model.to(device)
with profile(
    activities=[torch.profiler.ProfilerActivity.CPU],
    #on_trace_ready=torch.profiler.tensorboard_trace_handler("hf-prof"),
    profile_memory=True,
    with_stack=True,
    record_shapes=True,
) as prof:
    with torch.no_grad():
        #model.normalize()
        samples, labels = test_ld.dataset[0]

        samples = samples.to(device)  # keep batch dimension
        labels = labels

        samples_hv = encode(samples)
        outputs = model(samples_hv) #, dot=True)
        '''
        for samples, labels in tqdm(test_ld, desc="Testing"):
            samples = samples.to(device)

            samples_hv = encode(samples)
            outputs = model(samples_hv, dot=True)
        '''
output = prof.key_averages().table(
    sort_by="self_cpu_time_total",
    row_limit=50
)

with open("IntRVFLtime.txt", "w") as f:
    f.write(output)
