from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.profiler import profile, ProfilerActivity, record_function

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

torch.cuda.is_available = lambda: False

device = torch.device("cpu")

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

# Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint) #.to(device)
model.to(device)
model.eval()

raw_inputs = ["I've been waiting for a HuggingFace course my whole life."]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt") #.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

print("Model device:", next(model.parameters()).device)
print("Input device:", inputs["input_ids"].device)

with profile(
    activities=[torch.profiler.ProfilerActivity.CPU],
    #on_trace_ready=torch.profiler.tensorboard_trace_handler("hf-prof"),
    profile_memory=True,
    with_stack=True,
    record_shapes=True,
) as prof:
    print("here?")

    with torch.no_grad():
        outputs = model(**inputs)

print(outputs.logits)

output = prof.key_averages().table(
    sort_by="self_cpu_time_total",
    row_limit=50
)

with open("bertTime.txt", "w") as f:
    f.write(output)


#print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

