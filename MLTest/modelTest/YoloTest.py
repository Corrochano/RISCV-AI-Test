import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity, record_function

model = torch.load('../Newyolo11X.pt', weights_only=False)
inputs = torch.rand(1, 3, 640, 640)

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)

#print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=50))

output = prof.key_averages().table(
    sort_by="self_cpu_time_total",
    row_limit=50
)

with open("vectYolo11xTime.txt", "w") as f:
    f.write(output)
