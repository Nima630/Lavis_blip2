# # blip2_lavis_run.py
# import torch
# from PIL import Image
# from lavis.models import load_model_and_preprocess

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load model and preprocessors from LAVIS
# model, vis_processors, txt_processors = load_model_and_preprocess(
#     # name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device
#     # name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device
#     name="blip2_t5", model_type="instruct_flant5xl", is_eval=True, device=device


# )

# # Load image
# raw_image = Image.open("sss.jpg").convert("RGB")
# image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

# # Prepare prompts
# question = "what do the yellow lines represent?"
# question = "This is a technical image from a computer vision dataset. What do the yellow lines in the image indicate?"

# # Run model
# samples = {"image": image, "text_input": question}
# output = model.generate(samples)

# print("Answer:", output)




# run_blip2_infer.py — using Hugging Face BLIP-2 FLAN-T5-XL

import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image

# Choose device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load processor and model from Hugging Face
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl",
    device_map="auto",             # Automatically assigns layers to available GPU(s)
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)

# Load and prepare image
image = Image.open("sss.jpg").convert("RGB")

# List of questions you want to ask
questions = [
    "What do the yellow lines in the image represent?",
    "Is this image from a computer vision dataset?",
    "What kind of annotations are shown?",
    "What is the purpose of this image?"
]

# Loop through questions
for question in questions:
    # Prepare input
    inputs = processor(images=image, text=question, return_tensors="pt").to(device)

    # Generate answer
    output = model.generate(**inputs, max_new_tokens=50)
    answer = processor.decode(output[0], skip_special_tokens=True)

    # Print result
    print(f"Q: {question}\nA: {answer}\n")
