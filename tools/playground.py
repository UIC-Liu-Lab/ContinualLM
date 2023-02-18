from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    PretrainedConfig,
    get_scheduler,
)
from transformers import AutoTokenizer
from transformers import pipeline

# model_name_or_path = 'facebook/bart-large'
# model_name_or_path = 't5-large'
# model_name_or_path = 'EleutherAI/gpt-neo-1.3B'
# model_name_or_path = 'EleutherAI/gpt-j-6B'
model_name_or_path = 'EleutherAI/gpt-neo-2.7B'

print('model_name_or_path: ',model_name_or_path)

input_text = 'Who is the president of the States. Donald Trump. The reason is'

generator = pipeline('text-generation', model=model_name_or_path)

outputs = generator(input_text, do_sample=True, min_length=20, num_beams=2, num_return_sequences=2)

for output_id,output in enumerate(outputs):
    print('output_id: ',output_id)
    print(output['generated_text'])

# inputs = tokenizer(input_text, return_tensors="pt")
# summary_idss = model.generate(inputs["input_ids"], num_beams=10, min_length=0, max_length=20, num_return_sequences=10)
#
# for summary_ids in range(summary_idss.size(0)):
#     output = tokenizer.decode(summary_idss[summary_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False)
#     print(output)

