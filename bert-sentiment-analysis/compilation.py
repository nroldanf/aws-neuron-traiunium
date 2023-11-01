import torch
import torch_neuronx
import torch_xla.utils.serialization as xser
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

# https://huggingface.co/bert-base-cased
model_dir = "/opt/app/traiunium/bert-sentiment-analysis/models/checkpoints/checkpoint.pt"
model_name = "bert-base-cased"
max_length=128
# Setup some example inputs
sequence_0 = "The company HuggingFace is based in New York City"
sequence_1 = "Apples are especially bad for your health"
sequence_2 = "HuggingFace's headquarters are situated in Manhattan"

if __name__ == "__main__":
    # Build tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    state_dict = xser.load(model_dir)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_name, 
        state_dict=state_dict,
        num_labels=3, 
        return_dict=False, 
    )

    # ['input_ids', 'token_type_ids', 'attention_mask']
    paraphrase = tokenizer.encode_plus(sequence_0, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt")

    # Run the original PyTorch model on compilation exaple
    paraphrase_classification_logits = model(**paraphrase)[0]

    # Convert example inputs to a format that is compatible with TorchScript tracing
    example_inputs_paraphrase = paraphrase['input_ids'], paraphrase['attention_mask'], paraphrase['token_type_ids']

    # Run torch_neuronx.trace to generate a TorchScript that is optimized by AWS Neuron
    model_neuron = torch_neuronx.trace(model, example_inputs_paraphrase)
    
    # Save the TorchScript for later use
    model_neuron.save('models/neuron_compiled_model.pt')
    
