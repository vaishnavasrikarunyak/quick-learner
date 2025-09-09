# This script creates a personal finance chatbot using a Hugging Face model and Gradio.
# It is designed to be run locally in a Python environment.

import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# --- Model Loading and Configuration ---
# Use the specified model and tokenizer from Hugging Face.
# We'll load it with bfloat16 for memory efficiency if a GPU is available.
try:
    model_name = "ibm-granite/granite-3.3-2b-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map="auto" # This automatically manages device placement (GPU if available)
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")
    # Fallback to a simpler, non-model response if the model can't be loaded.
    def get_fallback_response(user_message, history):
        return "I'm sorry, I'm currently unable to provide financial advice. Please check your internet connection and model setup."
    gr.ChatInterface(get_fallback_response, title="Personal Finance Chatbot").launch()
    # Exit the script to prevent further errors
    import sys
    sys.exit()


# --- Chatbot Logic ---
# This function handles the interaction with the user and generates the chatbot's response.
def chat_with_granite(user_message, history):
    # A simple state to determine user demographic. This can be more sophisticated.
    # We'll use the history to check if the user has already identified themselves.
    is_demographic_set = False
    demographic = "general"
    for user_msg, bot_response in history:
        if "student" in user_msg.lower():
            demographic = "student"
            is_demographic_set = True
            break
        elif "professional" in user_msg.lower():
            demographic = "professional"
            is_demographic_set = True
            break
    
    # Initial greeting and demographic query
    if not is_demographic_set and "student" not in user_message.lower() and "professional" not in user_message.lower():
        return "Welcome to the Personal Finance Chatbot! To give you the best advice, please tell me if you are a *student* or a *professional*."
    
    # Craft a persona-based system prompt to guide the model's response
    if demographic == "student":
        system_prompt = (
            "You are a friendly and encouraging financial mentor for students. "
            "Your advice should be simple, easy to understand, and focused on core concepts like "
            "saving, budgeting, and understanding debt. Use a casual and approachable tone. "
            "Keep your responses concise and to the point."
        )
    elif demographic == "professional":
        system_prompt = (
            "You are a sophisticated and knowledgeable financial advisor for professionals. "
            "Your advice should be detailed, insightful, and cover topics like "
            "retirement planning (401k, Roth IRA), advanced investment strategies, and tax optimization. "
            "Use a formal and professional tone."
        )
    else: # Default case
        system_prompt = (
            "You are a general-purpose financial assistant. "
            "Provide helpful and neutral advice on savings, taxes, and investments. "
            "Keep your responses factual and clear."
        )

    # Combine the system prompt and the user's message.
    prompt = f"{system_prompt}\n\nUser: {user_message}\nAssistant:"

    # Generate a response from the model.
    try:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

        generated_tokens = model.generate(
            input_ids,
            max_new_tokens=200,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )

        output_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

        # Extract only the assistant's part of the conversation
        assistant_start_tag = "\nAssistant:"
        if assistant_start_tag in output_text:
            response = output_text.split(assistant_start_tag, 1)[-1].strip()
        else:
            response = output_text.strip()
            
        return response

    except Exception as e:
        print(f"Error during model generation: {e}")
        return "I'm sorry, I'm having trouble processing that request right now. Please try a different query."


# --- Gradio Interface Setup ---
# Create the Gradio ChatInterface.
iface = gr.ChatInterface(
    fn=chat_with_granite,
    chatbot=gr.Chatbot(height=500),
    textbox=gr.Textbox(
        placeholder="Ask me a question about savings, taxes, or investments...",
        container=False,
        scale=7
    ),
    title="Personal Finance Chatbot",
    description="I am an intelligent AI that provides tailored financial guidance. Let's get started!",
    theme="soft",
    examples=[
        "I am a student. What's a good way to start saving?",
        "I am a professional. How should I think about my 401(k) contributions?",
        "What is an IRA?"
    ]
)

# Launch the Gradio app.
if _name_ == "_main_":
    iface.launch()
