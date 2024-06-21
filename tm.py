import torch

# Check if CUDA is available and if a GPU is detected
if torch.cuda.is_available():
    print(f"Detected GPUs: {torch.cuda.device_count()}")
else:
    print("No GPU detected")
    
from openai import OpenAI
import json
from pygments import highlight
from pygments.lexers import JsonLexer
from pygments.formatters import TerminalFormatter
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
def to_dict(obj):
    return {attr: getattr(obj, attr) for attr in dir(obj) if not attr.startswith('__') and not callable(getattr(obj, attr))}


client = OpenAI(
    base_url='http://localhost:11434/v1/',

    # required but ignored
    api_key='ollama',
)

msg=[
        {
            'role': 'system',
            'content': 'you **must** produce JSON',
        },
        {
            'role': 'user',
            'content': 'Who is John Mbayu',
        }
    ]
chat_completion = client.chat.completions.create(
    messages=msg,
    model='llama3',
    logprobs=True,
    #response_format={ "type": "json_object" },    
)

encoding = tiktoken.encoding_for_model("llama3")
chat_completion_dict = to_dict(chat_completion)


#print(chat_completion_dict)

json_output = json.dumps(chat_completion_dict, indent=4, default=str)
# Colorize the JSON output
colorized_output = highlight(json_output, JsonLexer(), TerminalFormatter())

print(colorized_output)