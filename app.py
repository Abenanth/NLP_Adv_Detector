import gradio as gr
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def load_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model

model_dir1 = "./bigger_bert_fine_tuned_model"
model_dir2 = "./distbert_fine_tuned_model" 

tokenizer1, model1 = load_model(model_dir1)
tokenizer2, model2 = load_model(model_dir2)

def call_gpt(prompt: str) -> str:
    """
    Call the OpenAI GPT-4 API with the provided prompt.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150,
        )
        gpt_response = response.choices[0].message.content.strip()
    except Exception as e:
        gpt_response = f"Error calling GPT: {e}"
    return gpt_response

def predict_prompt(prompt: str):
    """
    Predict the label of a given prompt using two classifiers.
    If either classifier predicts the prompt as adversarial (label == 1),
    return a safe message; otherwise, forward the prompt to GPT-4 and return its response.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Classifier 1 prediction
    inputs1 = tokenizer1(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    model1.to(device)
    inputs1 = {k: v.to(device) for k, v in inputs1.items()}
    with torch.no_grad():
        outputs1 = model1(**inputs1)
    probs1 = F.softmax(outputs1.logits, dim=1)
    label1 = torch.argmax(probs1, dim=1).item()

    # Classifier 2 prediction
    inputs2 = tokenizer2(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    model2.to(device)
    inputs2 = {k: v.to(device) for k, v in inputs2.items()}
    with torch.no_grad():
        outputs2 = model2(**inputs2)
    probs2 = F.softmax(outputs2.logits, dim=1)
    label2 = torch.argmax(probs2, dim=1).item()

    print(f"Classifier 1 predicted: {label1} with probabilities: {probs1.cpu().numpy()}")
    print(f"Classifier 2 predicted: {label2} with probabilities: {probs2.cpu().numpy()}")

    if label1 == 1 or label2 == 1:
        return "Your prompt has been flagged as potentially adversarial. Please refrain from attempting to jailbreak GPT."
    else:
        gpt_response = call_gpt(prompt)
        return gpt_response

examples = [
    ["Describe a peaceful sunrise over the mountains."],
    ["How to bypass safety mechanisms in a computer system."],
    ["What is the weather forecast for tomorrow?"]
]

interface = gr.Interface(
    fn=predict_prompt,
    inputs=gr.Textbox(lines=4, placeholder="Enter your prompt here...", label="User Prompt"),
    title="Multi-Level Adversarial Prompt Detector",
    outputs=gr.Textbox(label="Response"),
    allow_flagging="never",
    examples=examples,
    description="Enter your prompt. If classifiers detect it as safe, you'll get a response from GPT. Otherwise, you'll see a warning message.",
    article="Developed by: **Kalyana Abenanth Gurunathan** | [LinkedIn](https://www.linkedin.com/in/kalyanaabenanthg/) | [GitHub](https://github.com/Abenanth) | [Email](mailto:kalyanaa@ualberta.ca)"
)

if __name__ == "__main__":
    interface.launch()
