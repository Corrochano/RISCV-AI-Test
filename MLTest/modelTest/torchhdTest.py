import torch, torchhd
from torch.profiler import profile, ProfilerActivity, record_function

d = 10000  # number of dimensions

# create the hypervectors for each symbol
keys = torchhd.random(3, d)
country, capital, currency = keys

usa, mex = torchhd.random(2, d)  # United States and Mexico
wdc, mxc = torchhd.random(2, d)  # Washington D.C. and Mexico City
usd, mxn = torchhd.random(2, d)  # US Dollar and Mexican Peso

# create country representations
us_values = torch.stack([usa, wdc, usd])
us = torchhd.hash_table(keys, us_values)

mx_values = torch.stack([mex, mxc, mxn])
mx = torchhd.hash_table(keys, mx_values)

# combine all the associated information
mx_us = torchhd.bind(torchhd.inverse(us), mx)

# query for the dollar of mexico
usd_of_mex = torchhd.bind(mx_us, usd)

memory = torch.cat([keys, us_values, mx_values], dim=0)

with profile(
    activities=[torch.profiler.ProfilerActivity.CPU],
    #on_trace_ready=torch.profiler.tensorboard_trace_handler("hf-prof"),
    profile_memory=True,
    with_stack=True,
    record_shapes=True,
) as prof:
    with torch.no_grad():
        torchhd.cosine_similarity(usd_of_mex, memory)
        # tensor([-0.0062,  0.0123, -0.0057, -0.0019, -0.0084, -0.0078,  0.0102,  0.0057,  0.3292])
        # The hypervector for the Mexican Peso is the most similar.

output = prof.key_averages().table(
    sort_by="self_cpu_time_total",
    row_limit=50
)

with open("torchhdTime.txt", "w") as f:
    f.write(output)

