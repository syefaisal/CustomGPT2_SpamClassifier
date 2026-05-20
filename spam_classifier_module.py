import torch
import tiktoken
from gptModule import GPTModel, generate_text_simple

# Initialize the model
CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])


print("Model configuration:", BASE_CONFIG)




GPT124M_classifier_Fintune_model = GPTModel(BASE_CONFIG)
# Modify output layer to length 2
torch.manual_seed(123)
num_classes=2
GPT124M_classifier_Fintune_model.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG["emb_dim"],
    out_features=num_classes
)
# print(GPT124M_classifier_Fintune_model)

# Load the model state dict
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
GPT124M_classifier_Fintune_model.to(device)
# Load the state dict with strict=False to ignore size mismatches
GPT124M_classifier_Fintune_model.load_state_dict(torch.load("LLM/review_classifier_llm_finetuned_Nov_01.pth", map_location=torch.device('mps'), weights_only=True), strict=False)

# Define the generate function
def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()

    # Prepare inputs the model
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]

    # Note: In the book, this was originally written as pos_emb.weight.shape[1] by mistake
    # It didn't break the code but would have caused unnecessary truncation (to 768 instead of 1024)

    #Truncate sequences if they are too long
    input_ids = input_ids[:min(max_length, supported_context_length)]

    # Pad sequences to the longest sequence
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)  # Add batch dimension

    # Model inference
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]  # Logits of last output token
    predicted_label = torch.argmax(logits, dim=-1).item()

    # return the classified result
    return "spam" if predicted_label == 1 else "not spam"


# Load the tokenizer
tokenizer  =tiktoken.get_encoding("gpt2")


# Define the classify function
def classify_review_response(query_text):
    return classify_review(
    query_text, GPT124M_classifier_Fintune_model, tokenizer, device, max_length=110)