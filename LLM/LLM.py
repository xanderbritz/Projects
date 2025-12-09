from transformers import pipeline, BitsAndBytesConfig
import torch

model_name = 'Microsoft/Phi-3-mini-4k-instruct'
print(f"Loading {model_name} in 4-bit...")

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Create pipeline with quantization
pipe = pipeline(
    "text-generation",
    model=model_name,
    model_kwargs={
        "quantization_config": bnb_config,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "max_memory": {0: "4.5GB"},
    },
    torch_dtype=torch.float16
)

print("Model loaded! Type 'quit' to exit or 'clear' to reset conversation.\n")
torch.cuda.empty_cache()

# Conversation history storage
conversation_history = []
MAX_HISTORY_TURNS = 5  # Keep last 5 exchanges to manage context length

def build_prompt(user_input):
    """Build a prompt with conversation history"""
    # Format conversation history
    context = ""
    for turn in conversation_history:
        context += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
    
    # Add current input
    context += f"User: {user_input}\nAssistant:"
    return context

while True:
    try:
        prompt = input('You: ').strip()
        
        if not prompt:
            continue
            
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        if prompt.lower() == 'clear':
            conversation_history = []
            print("Conversation history cleared!\n")
            continue
        
        # Build prompt with conversation context
        full_prompt = build_prompt(prompt)
        
        print("AI: ", end='', flush=True)
        result = pipe(
            full_prompt,
            max_new_tokens=30,
            temperature=0.5,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=2.0,
            no_repeat_ngram_size=3,
            return_full_text=False
        )
        
        generated_text = result[0]['generated_text'].strip()
        print(generated_text)
        
        # Store this exchange in history
        conversation_history.append({
            'user': prompt,
            'assistant': generated_text
        })
        
        # Trim history if it gets too long
        if len(conversation_history) > MAX_HISTORY_TURNS:
            conversation_history.pop(0)
        
        torch.cuda.empty_cache()
        
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        break
    except torch.cuda.OutOfMemoryError:
        print("\nOut of memory! Try shorter prompts or type 'clear' to reset history.")
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"\nError: {e}")