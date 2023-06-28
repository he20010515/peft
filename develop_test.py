from transformers import AutoModel,AutoTokenizer
import torch
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from peft import PrototypeLoraModel, PrototypeLoraConfig,PeftModel
import json
model_name_or_path = "t5-base"
tokenizer_name_or_path = "t5-base"












peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,inference_mode=False,r=4,lora_alpha=32,lora_dropout=0.1,
)
model = AutoModel.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
p = model.all_parameters()
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
input_ids = tokenizer.encode("this is a test input")
print("Input:", tokenizer.decode(input_ids))
input_ids = tokenizer(
    "Studies have been shown that owning a dog is good for you", return_tensors="pt"
).input_ids  # Batch size 1
decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
# preprocess: Prepend decoder_input_ids with start token which is pad token for T5Model.
# This is not needed for torch's T5ForConditionalGeneration as it does this internally using labels arg.
decoder_input_ids = model._shift_right(decoder_input_ids)

# forward pass
outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
# print("Output shape:",outputs)

# output: trainable params: 2359296 || all params: 1231940608 || trainable%: 0.19151053100118282
with open("lora.p.json",'w') as f:   
    json.dump(p,f)





peft_config = PrototypeLoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,inference_mode=False,r=4,lora_alpha=32,lora_dropout=0.1,sparsity=0.1
)
model = AutoModel.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
p = model.all_parameters()
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
input_ids = tokenizer.encode("this is a test input")
print("Input:", tokenizer.decode(input_ids))
input_ids = tokenizer(
    "Studies have been shown that owning a dog is good for you", return_tensors="pt"
).input_ids  # Batch size 1
decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
# preprocess: Prepend decoder_input_ids with start token which is pad token for T5Model.
# This is not needed for torch's T5ForConditionalGeneration as it does this internally using labels arg.
decoder_input_ids = model._shift_right(decoder_input_ids)

# forward pass
outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
# print("Output shape:",outputs)

# output: trainable params: 2359296 || all params: 1231940608 || trainable%: 0.19151053100118282
with open("protypelora.p.json",'w') as f:   
    json.dump(p,f)



model = AutoModel.from_pretrained(model_name_or_path)
p = PeftModel.all_parameters(model)

with open("original.t5.json",'w') as f:   
    json.dump(p,f)
