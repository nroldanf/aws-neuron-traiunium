import torch
import torch_neuronx
import torch_xla.utils.serialization as xser
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

model_dir = "/opt/app/traiunium/bert-sentiment-analysis/models/checkpoint.pt"
max_length=128
# Setup some example inputs
sequence_0 = "The company HuggingFace is based in New York City"
sequence_1 = "Apples are especially bad for your health"
sequence_2 = "HuggingFace's headquarters are situated in Manhattan"

if __name__ == "__main__":
    # Build tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    state_dict = xser.load(model_dir)
    
    # model = torch.load(model_dir).keys()
    
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     pretrained_model_name_or_path=model_dir, 
    #     num_labels=3, 
    #     return_dict=False, 
    #     from_pt=True
    # )

    # paraphrase = tokenizer.encode_plus(sequence_0, sequence_2, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt")
    # not_paraphrase = tokenizer.encode_plus(sequence_0, sequence_1, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt")

    # # Run the original PyTorch model on compilation exaple
    # paraphrase_classification_logits = model(**paraphrase)[0]

    # # Convert example inputs to a format that is compatible with TorchScript tracing
    # example_inputs_paraphrase = paraphrase['input_ids'], paraphrase['attention_mask'], paraphrase['token_type_ids']
    # example_inputs_not_paraphrase = not_paraphrase['input_ids'], not_paraphrase['attention_mask'], not_paraphrase['token_type_ids']

    # # Run torch_neuronx.trace to generate a TorchScript that is optimized by AWS Neuron
    # model_neuron = torch_neuronx.trace(model, example_inputs_paraphrase)
    
    # # Save the TorchScript for later use
    # model_neuron.save('models/neuron_compiled_model.pt')
    
