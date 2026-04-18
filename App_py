import streamlit as st
import torch
from diffusers import StableDiffusionPipeline

st.set_page_config(page_title="Text to Image Generator", layout="centered")

st.title("🎨 Text → Image Generator")
st.write("Enter a prompt and generate an image using Stable Diffusion")

@st.cache_resource
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")

    return pipe

pipe = load_model()

prompt = st.text_input("Enter your prompt", "A futuristic city at sunset")

if st.button("Generate Image"):
    with st.spinner("Generating... please wait"):
        image = pipe(prompt).images[0]
        st.image(image, caption=prompt, use_container_width=True)
