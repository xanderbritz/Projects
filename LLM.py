from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = 'gpt2-xl'
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)


if torch.cuda.is_available():
    model = model.to('cuda')
    print("Model loaded on GPU!")
else:# Move to GPU if available
    print("Model loaded on CPU (this will be slower)")

print("Ready! Type 'quit' to exit.\n")

while True:
    try:
        prompt = input('You: ').strip()
        
        if not prompt:
            continue
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        print("AI: ", end='', flush=True)
        inputs = tokenizer(prompt, return_tensors='pt')
        
        # Move inputs to same de# Move to GPU if available# Move to GPU if availablevice as model
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100,
            temperature=0.5,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=2.0,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id
        )
        
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = full_text[len(prompt):].strip()
        print(generated_text)
        
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        break
    except Exception as e:
        print(f"\nError: {e}")