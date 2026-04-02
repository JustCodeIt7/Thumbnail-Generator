import io
import json
import os
import textwrap
import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from PIL import ImageDraw, ImageFont, ImageStat

############################# Constants & Config ###############################

load_dotenv()
TRANSCRIPT_LIMIT = 12000
IMAGE_SIZE = (1024, 576)
IDEA_KEYS = ("headline", "hook", "visual", "style", "prompt")
# prompthero/openjourney stabilityai/sd-turbo
DEFAULT_IMAGE_MODEL = os.getenv("IMAGE_MODEL_ID", "stabilityai/sd-turbo")


############################ Device Detection ##################################


def image_device():
    """Determine the best available compute device for image generation."""
    # Allow manual override via environment variable
    if os.getenv("IMAGE_DEVICE"):
        return os.getenv("IMAGE_DEVICE")
    import torch

    # Prefer Apple Silicon GPU, then CUDA, then fall back to CPU
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


######################### Diffusion Pipeline Loader ############################


@st.cache_resource(show_spinner=False)
def load_pipeline(model_id, device):
    """Load and cache a text-to-image diffusion pipeline on the target device."""
    import torch
    from diffusers import AutoPipelineForText2Image

    # Use float16 for CUDA to save VRAM; float32 for CPU/MPS compatibility
    dtype = torch.float16 if device == "cuda" else torch.float32
    kwargs = {"torch_dtype": dtype, "use_safetensors": True}
    if device == "cuda":
        kwargs["variant"] = "fp16"
    # Fall back without fp16 variant if the model doesn't ship one
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(model_id, **kwargs)
    except Exception:
        kwargs.pop("variant", None)
        pipe = AutoPipelineForText2Image.from_pretrained(model_id, **kwargs)
    pipe = pipe.to(device)
    # Reduce memory pressure on Apple Silicon
    if device == "mps":
        pipe.enable_attention_slicing()
    return pipe


########################### Image Utilities ####################################


def is_blank(image):
    """Return True if the image is effectively empty (all black or no content)."""
    gray = image.convert("L")
    return gray.getbbox() is None or ImageStat.Stat(gray).mean[0] < 3


######################### LLM Response Parsing ################################


def parse_ideas(raw, count):
    """Parse the LLM's raw text into a list of validated thumbnail idea dicts."""
    text = raw.strip()
    # Strip markdown code fences if the LLM wrapped its response
    if "```" in text:
        text = (
            text.split("```")[1 if text.startswith("```") else 0]
            .replace("json", "", 1)
            .strip()
        )
    # Extract the JSON array from potentially noisy output
    start, end = text.find("["), text.rfind("]")
    ideas = json.loads(text[start : end + 1]) if start >= 0 and end > start else []
    # Normalize each idea to only the expected keys
    cleaned = [
        {k: str(item.get(k, "")).strip() for k in IDEA_KEYS} for item in ideas[:count]
    ]
    if len(cleaned) != count or any(not idea["headline"] for idea in cleaned):
        raise ValueError("Planner returned incomplete thumbnail ideas.")
    return cleaned


########################## Thumbnail Planning ##################################


def plan_thumbnails(transcript, count):
    """Send the transcript to Ollama and return a list of thumbnail concepts."""
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
    # Invoke Ollama with a higher temperature for creative variety
    try:
        raw = (
            ChatOllama(
                model=os.getenv("OLLAMA_MODEL", "llama3.2:latest"), temperature=0.8
            )
            .invoke(prompt)
            .content
        )
    except Exception as e:
        raise RuntimeError(
            "Ollama planning failed. Start Ollama and pull the configured model first."
        ) from e
    return parse_ideas(raw if isinstance(raw, str) else json.dumps(raw), count)


############################ Font Loading ######################################


def load_font(size):
    """Load a bold system font, falling back through common paths to the default."""
    for path in [
        os.getenv("FONT_PATH"),
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Impact.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ]:
        if path and os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


######################### Headline Overlay #####################################


def add_headline(image, headline):
    """Overlay a bold, centered headline with a semi-transparent backdrop."""
    image = image.convert("RGBA")
    draw = ImageDraw.Draw(image)
    # Scale font size relative to image width
    font = load_font(max(42, image.width // 13))
    text = textwrap.fill(headline.upper(), width=14)
    # Measure the text bounding box to center it
    left, top, right, bottom = draw.multiline_textbbox(
        (0, 0), text, font=font, spacing=6, stroke_width=4
    )
    pad = 24
    x = (image.width - (right - left)) // 2
    y = image.height - (bottom - top) - pad * 2
    # Draw a dark rounded rectangle behind the text for readability
    draw.rounded_rectangle(
        (pad, y - pad, image.width - pad, image.height - pad // 2),
        radius=18,
        fill=(0, 0, 0, 150),  # Semi-transparent black
    )
    # Render the headline with a black stroke for contrast
    draw.multiline_text(
        (x, y),
        text,
        font=font,
        fill="white",
        align="center",
        spacing=6,
        stroke_width=4,
        stroke_fill="black",
    )
    return image.convert("RGB")


######################### Thumbnail Rendering ##################################


def render_thumbnail(idea, used):
    """Generate a thumbnail image from an idea dict using the local diffusion model."""
    model_id = DEFAULT_IMAGE_MODEL
    # Build a detailed prompt for the diffusion model
    prompt = f"""
    Create a polished, bold YouTube thumbnail background in 16:9.
    Main visual: {idea["visual"]}
    Style: {idea["style"]}
    Click hook: {idea["hook"]}
    Additional direction: {idea["prompt"]}
    Avoid text, letters, logos, watermarks, clutter, and repeated ideas.
    Do not copy these previous headlines: {", ".join(used) or "none"}
    """.strip()
    # Turbo models use fewer steps and zero guidance for speed
    turbo = "turbo" in model_id.lower()
    params = {
        "prompt": prompt,
        "negative_prompt": "text, letters, words, logo, watermark, blurry, clutter",
        "width": IMAGE_SIZE[0],
        "height": IMAGE_SIZE[1],
        "num_inference_steps": 4 if turbo else 24,
        "guidance_scale": 0.0 if turbo else 7.5,
    }
    # Attempt generation on preferred device, fall back to CPU if MPS produces blanks
    try:
        device = image_device()
        image = load_pipeline(model_id, device)(**params).images[0]
        if is_blank(image) and device == "mps":
            image = load_pipeline(model_id, "cpu")(**params).images[0]
        if is_blank(image):
            raise RuntimeError("The local model returned a blank image.")
    except Exception as e:
        raise RuntimeError(
            "Local image generation failed. Check diffusers installs, model availability, and device memory."
        ) from e
    # Composite the headline text onto the generated image
    image = add_headline(image, idea["headline"])
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return {**idea, "prompt": prompt, "png": buf.getvalue()}


########################### Streamlit UI #######################################
# TODO Rename this here and in `main`
def generate_thumbnails(transcript, count):
    # Step 1: Use Ollama to brainstorm thumbnail concepts
    with st.spinner("Planning thumbnail concepts with Ollama..."):
        ideas = plan_thumbnails(transcript.strip(), int(count))
    # Step 2: Render each concept through the diffusion pipeline
    progress = st.progress(0, text="Generating thumbnails locally...")
    images, used = [], []
    for i, idea in enumerate(ideas, start=1):
        images.append(render_thumbnail(idea, used))
        used.append(idea["headline"])  # Track used headlines to avoid repeats
        progress.progress(
            i / len(ideas), text=f"Generating thumbnail {i}/{len(ideas)}..."
        )
    progress.empty()

    # Display the creative rationale behind each thumbnail
    st.subheader("Concepts")
    for i, item in enumerate(images, start=1):
        with st.expander(f"#{i} - {item['headline']}", expanded=True):
            st.write(f"**Hook:** {item['hook']}")
            st.write(f"**Visual:** {item['visual']}")
            st.code(item["prompt"], language="text")
    # Render final thumbnails in a 3-column grid with download buttons
    st.subheader("Generated thumbnails")
    cols = st.columns(3)
    for i, item in enumerate(images):
        with cols[i % 3]:
            st.image(item["png"], caption=item["headline"], width="stretch")
            st.download_button(
                f"Download #{i + 1}",
                data=item["png"],
                file_name=f"thumbnail_{i + 1}.png",
                mime="image/png",
                width="stretch",
            )


def render_sidebar():
    """Render the sidebar with configuration controls and model info."""
    with st.sidebar:
        count = st.number_input(
            "How many thumbnails?", min_value=1, max_value=6, value=3, step=1
        )
        st.text_input(
            "Ollama model",
            value=os.getenv("OLLAMA_MODEL", "llama3.2:latest"),
            disabled=True,
        )
        st.text_input(
            "Image model",
            value=DEFAULT_IMAGE_MODEL,
            disabled=True,
        )
        st.caption(f"Image device: `{image_device()}`")
        st.caption("First run may download model weights from Hugging Face.")
    return count


def main():
    """Main Streamlit app function to orchestrate the thumbnail generation workflow."""
    st.set_page_config(
        page_title="Local Thumbnail Generator", page_icon="🎬", layout="wide"
    )
    st.title("🎬 Local YouTube Thumbnail Generator")
    st.caption(
        "Plan with Ollama, render with a local diffusion model, then add readable headline text in-app."
    )

    # Sidebar: configuration controls (read-only display of active models)
    count = render_sidebar()
    # Main input area for the video transcript
    transcript = st.text_area(
        "YouTube video transcript",
        height=280,
        placeholder="Paste the full transcript here...",
    )

    ######################### Generation Workflow ##################################

    if st.button("Generate thumbnails", type="primary", width="stretch"):
        if not transcript.strip():
            st.error("Please paste a transcript first.")
        else:
            generate_thumbnails(transcript, count)


if __name__ == "__main__":
    main()
