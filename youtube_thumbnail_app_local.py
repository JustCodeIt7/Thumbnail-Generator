import io
import json
import os
import textwrap
import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from PIL import ImageDraw, ImageFont

load_dotenv()
TRANSCRIPT_LIMIT = 12000
IMAGE_SIZE = (1024, 576)
IDEA_KEYS = ("headline", "hook", "visual", "style", "prompt")

def image_device():
    if os.getenv("IMAGE_DEVICE"):
        return os.getenv("IMAGE_DEVICE")
    import torch
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource(show_spinner=False)
def load_pipeline(model_id, device):
    try:
        import torch
        from diffusers import AutoPipelineForText2Image
    except Exception as e:
        raise RuntimeError("Missing local image packages. Run: pip install diffusers transformers accelerate safetensors") from e
    dtype = torch.float16 if device != "cpu" else torch.float32
    kwargs = {"torch_dtype": dtype, "use_safetensors": True}
    if dtype == torch.float16:
        kwargs["variant"] = "fp16"
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(model_id, **kwargs)
    except Exception:
        kwargs.pop("variant", None)
        pipe = AutoPipelineForText2Image.from_pretrained(model_id, **kwargs)
    pipe = pipe.to(device)
    if device == "mps":
        pipe.enable_attention_slicing()
    return pipe

def parse_ideas(raw, count):
    text = raw.strip()
    if "```" in text:
        text = text.split("```")[1 if text.startswith("```") else 0].replace("json", "", 1).strip()
    start, end = text.find("["), text.rfind("]")
    ideas = json.loads(text[start : end + 1]) if start >= 0 and end > start else []
    cleaned = [{k: str(item.get(k, "")).strip() for k in IDEA_KEYS} for item in ideas[:count]]
    if len(cleaned) != count or any(not idea["headline"] for idea in cleaned):
        raise ValueError("Planner returned incomplete thumbnail ideas.")
    return cleaned

def plan_thumbnails(transcript, count):
    prompt = f"""
You design highly clickable YouTube thumbnails.
Return ONLY a JSON array with exactly {count} objects.
Each object must contain: headline, hook, visual, style, prompt.
Rules:
- headline is 2-5 words
- every concept is visually distinct
- optimize for CTR and dramatic contrast
- no markdown, no prose, no code fences
Transcript:
{transcript[:TRANSCRIPT_LIMIT]}
""".strip()
    try:
        raw = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.2:latest"), temperature=0.8).invoke(prompt).content
    except Exception as e:
        raise RuntimeError("Ollama planning failed. Start Ollama and pull the configured model first.") from e
    return parse_ideas(raw if isinstance(raw, str) else json.dumps(raw), count)

def load_font(size):
    for path in [
        os.getenv("FONT_PATH"),
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Impact.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ]:
        if path and os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()

def add_headline(image, headline):
    image = image.convert("RGBA")
    draw = ImageDraw.Draw(image)
    font = load_font(max(42, image.width // 13))
    text = textwrap.fill(headline.upper(), width=14)
    left, top, right, bottom = draw.multiline_textbbox((0, 0), text, font=font, spacing=6, stroke_width=4)
    pad = 24
    x = (image.width - (right - left)) // 2
    y = image.height - (bottom - top) - pad * 2
    draw.rounded_rectangle((pad, y - pad, image.width - pad, image.height - pad // 2), radius=18, fill=(0, 0, 0, 150))
    draw.multiline_text((x, y), text, font=font, fill="white", align="center", spacing=6, stroke_width=4, stroke_fill="black")
    return image.convert("RGB")

def render_thumbnail(idea, used):
    model_id = os.getenv("IMAGE_MODEL_ID", "stabilityai/sd-turbo")
    prompt = f"""
Create a polished, bold YouTube thumbnail background in 16:9.
Main visual: {idea['visual']}
Style: {idea['style']}
Click hook: {idea['hook']}
Additional direction: {idea['prompt']}
Avoid text, letters, logos, watermarks, clutter, and repeated ideas.
Do not copy these previous headlines: {", ".join(used) or "none"}
""".strip()
    turbo = "turbo" in model_id.lower()
    try:
        image = load_pipeline(model_id, image_device())(
            prompt=prompt,
            negative_prompt="text, letters, words, logo, watermark, blurry, clutter",
            width=IMAGE_SIZE[0],
            height=IMAGE_SIZE[1],
            num_inference_steps=4 if turbo else 24,
            guidance_scale=0.0 if turbo else 7.5,
        ).images[0]
    except Exception as e:
        raise RuntimeError("Local image generation failed. Check diffusers installs, model availability, and device memory.") from e
    image = add_headline(image, idea["headline"])
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return {**idea, "prompt": prompt, "png": buf.getvalue()}

st.set_page_config(page_title="Local Thumbnail Generator", page_icon="🎬", layout="wide")
st.title("🎬 Local YouTube Thumbnail Generator")
st.caption("Plan with Ollama, render with a local diffusion model, then add readable headline text in-app.")
with st.sidebar:
    count = st.number_input("How many thumbnails?", min_value=1, max_value=6, value=3, step=1)
    st.text_input("Ollama model", value=os.getenv("OLLAMA_MODEL", "llama3.2:latest"), disabled=True)
    st.text_input("Image model", value=os.getenv("IMAGE_MODEL_ID", "stabilityai/sd-turbo"), disabled=True)
    st.caption(f"Image device: `{image_device()}`")
    st.caption("First run may download model weights from Hugging Face.")
transcript = st.text_area("YouTube video transcript", height=280, placeholder="Paste the full transcript here...")

if st.button("Generate thumbnails", type="primary", use_container_width=True):
    if not transcript.strip():
        st.error("Please paste a transcript first.")
    else:
        try:
            with st.spinner("Planning thumbnail concepts with Ollama..."):
                ideas = plan_thumbnails(transcript.strip(), int(count))
            progress = st.progress(0, text="Generating thumbnails locally...")
            images, used = [], []
            for i, idea in enumerate(ideas, start=1):
                images.append(render_thumbnail(idea, used))
                used.append(idea["headline"])
                progress.progress(i / len(ideas), text=f"Generating thumbnail {i}/{len(ideas)}...")
            progress.empty()
        except Exception as e:
            st.error(str(e))
        else:
            st.subheader("Concepts")
            for i, item in enumerate(images, start=1):
                with st.expander(f"#{i} - {item['headline']}", expanded=True):
                    st.write(f"**Hook:** {item['hook']}")
                    st.write(f"**Visual:** {item['visual']}")
                    st.code(item["prompt"], language="text")
            st.subheader("Generated thumbnails")
            cols = st.columns(3)
            for i, item in enumerate(images):
                with cols[i % 3]:
                    st.image(item["png"], caption=item["headline"], use_container_width=True)
                    st.download_button(f"Download #{i + 1}", data=item["png"], file_name=f"thumbnail_{i + 1}.png", mime="image/png", use_container_width=True)
