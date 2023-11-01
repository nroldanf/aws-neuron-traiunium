import os
import json
import torch
import torch_neuronx
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

model_name = "bert-base-cased"
model_dir = "/opt/app/traiunium/bert-sentiment-analysis/models/neuron_compiled_model.pt"
JSON_CONTENT_TYPE = 'application/json'

def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    if content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(serialized_input_data)
        return input_data
    else:
        raise Exception('Requested unsupported ContentType in Accept: ' + content_type)
        return

def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)

def model_fn(model_dir):
    tokenizer_init = AutoTokenizer.from_pretrained(model_name)
    model_file = os.path.join(model_dir, model_dir)
    model_neuron = torch.jit.load(model_file)
    return (model_neuron, tokenizer_init)

def predict_fn(input_data, models):
    model_bert, tokenizer = models
    sequence_0 = input_data[0] 
    
    max_length=128
    # max_length=max_length, padding='max_length', truncation=True,
    paraphrase = tokenizer.encode_plus(
        sequence_0, 
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors="pt",
    )
    # Convert example inputs to a format that is compatible with TorchScript tracing
    example_inputs = paraphrase['input_ids'], paraphrase['attention_mask'], paraphrase['token_type_ids']  

    # Verify the TorchScript works on example inputs
    classification_logits_neuron = model_bert(*example_inputs)
    classes = ['neutral', 'positive', 'negative']
    prediction = classification_logits_neuron[0][0].argmax().item()
    out_str = 'BERT says that "{}" is {}'.format(sequence_0, classes[prediction])
    
    return out_str

if __name__ == "__main__":
    inputs = ["Metal Gear Solid master collection is so bad"]
    model = model_fn(model_dir)
    prediction = predict_fn(inputs, model)
    print(prediction)