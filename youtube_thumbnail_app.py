import os
import base64
from typing import List, TypedDict

import streamlit as st
from pydantic import BaseModel, Field
from openai import OpenAI
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from rich import print


############################# Data Models #############################


# Define structured output schema for a single thumbnail concept
class ThumbnailIdea(BaseModel):
    headline: str = Field(description="Short overlay text, 2-5 words")
    hook: str = Field(description="Why this thumbnail is clickable")
    visual: str = Field(description="Main visual scene or composition")
    style: str = Field(description="Art direction and color mood")
    prompt: str = Field(description="Detailed image prompt for the thumbnail")


# Wrap multiple ideas into one response object for structured parsing
class ThumbnailPlan(BaseModel):
    ideas: List[ThumbnailIdea] = Field(description="List of unique thumbnail concepts")


# Track data flowing through the LangGraph pipeline
class AppState(TypedDict):
    transcript: str
    count: int
    ideas: list
    images: list


# Cap transcript length to stay within token limits
TRANSCRIPT_LIMIT = 12000


######################### Model Initialization #########################


def get_text_model() -> ChatOpenAI:
    """Return a ChatOpenAI instance configured for creative thumbnail ideation."""
    return ChatOpenAI(
        model=os.getenv("TEXT_MODEL", "gpt-4.1-mini"),
        temperature=0.9,  # High temperature for diverse creative output
    )


def get_image_client() -> OpenAI:
    """Return a raw OpenAI client for image generation."""
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


########################## Graph Node: Plan ############################


def plan_thumbnails(state: AppState):
    """Generate structured thumbnail concepts from a video transcript using an LLM."""
    # Bind the model to return a validated ThumbnailPlan object
    planner = get_text_model().with_structured_output(
        ThumbnailPlan, method="json_schema"
    )
    transcript = state["transcript"][:TRANSCRIPT_LIMIT]
    prompt = f"""
You design highly clickable YouTube thumbnails.

Create EXACTLY {state["count"]} unique thumbnail concepts from the transcript below.
Each idea must feel visually different from the others.

Rules:
- Optimize for YouTube CTR.
- Use short, punchy overlay text.
- Prefer one dominant subject and one clear visual story.
- Avoid generic stock-photo vibes.
- Make each idea distinct in angle, framing, and visual metaphor.
- The final image should be landscape 16:9.

Transcript:
{transcript}
""".strip()
    result = planner.invoke(prompt)
    # Enforce the requested count and convert to plain dicts for state serialization
    ideas = [i.model_dump() for i in result.ideas[: state["count"]]]
    return {"ideas": ideas}


######################### Graph Node: Render ###########################


def render_thumbnails(state: AppState):
    """Generate actual thumbnail images for each planned concept via the OpenAI image API."""
    client = get_image_client()
    outputs = []
    used_heads = []
    # Generate one image per idea, injecting prior headlines to encourage diversity
    for idx, idea in enumerate(state["ideas"], start=1):
        diversity_note = (
            "Previous headlines to avoid repeating: " + ", ".join(used_heads)
            if used_heads
            else ""
        )
        img_prompt = f"""
Create a polished, bold YouTube thumbnail in 16:9.

Headline text in image: {idea["headline"]}
Click hook: {idea["hook"]}
Main visual: {idea["visual"]}
Style: {idea["style"]}
Additional direction: {idea["prompt"]}

Constraints:
- One strong focal point
- Large readable text
- High contrast and dramatic composition
- Minimal clutter
- No watermark
- Distinct from the other thumbnails in this batch
- Clean professional thumbnail design
{diversity_note}
""".strip()
        res = client.images.generate(
            model=os.getenv("IMAGE_MODEL", "gpt-image-1.5"),
            prompt=img_prompt,
            size="1536x1024",  # 16:9 landscape ratio
            quality=os.getenv("IMAGE_QUALITY", "medium"),
            output_format="png",
        )
        outputs.append(
            {
                "index": idx,
                "headline": idea["headline"],
                "hook": idea["hook"],
                "visual": idea["visual"],
                "prompt": img_prompt,
                "b64": res.data[0].b64_json,  # Raw base64 PNG for display and download
            }
        )
        # Track used headlines so the next iteration can avoid repetition
        used_heads.append(idea["headline"])
    return {"images": outputs}


######################### LangGraph Pipeline ###########################


def build_graph():
    """Compile a two-step LangGraph: plan concepts then render images."""
    graph = StateGraph(AppState)
    graph.add_node("plan", plan_thumbnails)
    graph.add_node("render", render_thumbnails)
    graph.set_entry_point("plan")
    graph.add_edge("plan", "render")
    graph.add_edge("render", END)
    return graph.compile()


########################### Streamlit UI ################################

st.set_page_config(page_title="AI Thumbnail Generator", page_icon="🎬", layout="wide")
st.title("🎬 AI YouTube Thumbnail Generator")
st.caption(
    "Paste a transcript, choose a count, and generate unique thumbnail concepts + images."
)

# Render sidebar with configuration inputs
with st.sidebar:
    st.header("Settings")
    count = st.number_input(
        "How many thumbnails?", min_value=1, max_value=6, value=3, step=1
    )
    st.text_input(
        "Text model", value=os.getenv("TEXT_MODEL", "gpt-4.1-mini"), disabled=True
    )
    st.text_input(
        "Image model", value=os.getenv("IMAGE_MODEL", "gpt-image-1.5"), disabled=True
    )
    st.markdown("Set `OPENAI_API_KEY` before running the app.")

transcript = st.text_area(
    "YouTube video transcript",
    height=280,
    placeholder="Paste the full transcript here...",
)

generate = st.button("Generate thumbnails", type="primary", width="stretch")

######################### Generation Handler ############################

# Validate inputs then run the full plan → render pipeline
if generate:
    graph = build_graph()
    progress = st.progress(0, text="Planning thumbnail concepts...")
    result = graph.invoke(
        {
            "transcript": transcript.strip(),
            "count": int(count),
            "ideas": [],
            "images": [],
        }
    )
    progress.progress(100, text="Done")

        # Display each concept's details in expandable sections
        st.subheader("Concepts")
        for item in result["images"]:
            with st.expander(f"#{item['index']} — {item['headline']}", expanded=True):
                st.write(f"**Hook:** {item['hook']}")
                st.write(f"**Visual:** {item['visual']}")
                st.code(item["prompt"], language="text")

        # Render thumbnails in a 3-column grid with download buttons
        st.subheader("Generated thumbnails")
        cols = st.columns(3)
        for i, item in enumerate(result["images"]):
            img_bytes = base64.b64decode(item["b64"])
            with cols[i % 3]:  # Cycle through columns for even distribution
                st.image(img_bytes, caption=item["headline"], width="content")
                st.download_button(
                    label=f"Download #{item['index']}",
                    data=img_bytes,
                    file_name=f"thumbnail_{item['index']}.png",
                    mime="image/png",
                    use_container_width=True,
                )
