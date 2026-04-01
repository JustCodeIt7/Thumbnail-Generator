import os
import base64
from typing import List, TypedDict

import streamlit as st
from pydantic import BaseModel, Field
from openai import OpenAI
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI


class ThumbnailIdea(BaseModel):
    headline: str = Field(description="Short overlay text, 2-5 words")
    hook: str = Field(description="Why this thumbnail is clickable")
    visual: str = Field(description="Main visual scene or composition")
    style: str = Field(description="Art direction and color mood")
    prompt: str = Field(description="Detailed image prompt for the thumbnail")


class ThumbnailPlan(BaseModel):
    ideas: List[ThumbnailIdea] = Field(description="List of unique thumbnail concepts")


class AppState(TypedDict):
    transcript: str
    count: int
    ideas: list
    images: list


TRANSCRIPT_LIMIT = 12000


def get_text_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("TEXT_MODEL", "gpt-4.1-mini"),
        temperature=0.9,
    )


def get_image_client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def plan_thumbnails(state: AppState):
    planner = get_text_model().with_structured_output(ThumbnailPlan, method="json_schema")
    transcript = state["transcript"][:TRANSCRIPT_LIMIT]
    prompt = f"""
You design highly clickable YouTube thumbnails.

Create EXACTLY {state['count']} unique thumbnail concepts from the transcript below.
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
    ideas = [i.model_dump() for i in result.ideas[: state["count"]]]
    if len(ideas) < state["count"]:
        raise ValueError("Model returned fewer thumbnail ideas than requested.")
    return {"ideas": ideas}


def render_thumbnails(state: AppState):
    client = get_image_client()
    outputs = []
    used_heads = []
    for idx, idea in enumerate(state["ideas"], start=1):
        diversity_note = "Previous headlines to avoid repeating: " + ", ".join(used_heads) if used_heads else ""
        img_prompt = f"""
Create a polished, bold YouTube thumbnail in 16:9.

Headline text in image: {idea['headline']}
Click hook: {idea['hook']}
Main visual: {idea['visual']}
Style: {idea['style']}
Additional direction: {idea['prompt']}

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
            size="1536x1024",
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
                "b64": res.data[0].b64_json,
            }
        )
        used_heads.append(idea["headline"])
    return {"images": outputs}


def build_graph():
    graph = StateGraph(AppState)
    graph.add_node("plan", plan_thumbnails)
    graph.add_node("render", render_thumbnails)
    graph.set_entry_point("plan")
    graph.add_edge("plan", "render")
    graph.add_edge("render", END)
    return graph.compile()


st.set_page_config(page_title="AI Thumbnail Generator", page_icon="🎬", layout="wide")
st.title("🎬 AI YouTube Thumbnail Generator")
st.caption("Paste a transcript, choose a count, and generate unique thumbnail concepts + images.")

with st.sidebar:
    st.header("Settings")
    count = st.number_input("How many thumbnails?", min_value=1, max_value=6, value=3, step=1)
    st.text_input("Text model", value=os.getenv("TEXT_MODEL", "gpt-4.1-mini"), disabled=True)
    st.text_input("Image model", value=os.getenv("IMAGE_MODEL", "gpt-image-1.5"), disabled=True)
    st.markdown("Set `OPENAI_API_KEY` before running the app.")

transcript = st.text_area(
    "YouTube video transcript",
    height=280,
    placeholder="Paste the full transcript here...",
)

generate = st.button("Generate thumbnails", type="primary", use_container_width=True)

if generate:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Missing OPENAI_API_KEY environment variable.")
    elif not transcript.strip():
        st.error("Please paste a transcript first.")
    else:
        graph = build_graph()
        progress = st.progress(0, text="Planning thumbnail concepts...")
        try:
            result = graph.invoke(
                {"transcript": transcript.strip(), "count": int(count), "ideas": [], "images": []}
            )
            progress.progress(100, text="Done")

            st.subheader("Concepts")
            for item in result["images"]:
                with st.expander(f"#{item['index']} — {item['headline']}", expanded=True):
                    st.write(f"**Hook:** {item['hook']}")
                    st.write(f"**Visual:** {item['visual']}")
                    st.code(item["prompt"], language="text")

            st.subheader("Generated thumbnails")
            cols = st.columns(3)
            for i, item in enumerate(result["images"]):
                img_bytes = base64.b64decode(item["b64"])
                with cols[i % 3]:
                    st.image(img_bytes, caption=item["headline"], use_container_width=True)
                    st.download_button(
                        label=f"Download #{item['index']}",
                        data=img_bytes,
                        file_name=f"thumbnail_{item['index']}.png",
                        mime="image/png",
                        use_container_width=True,
                    )
        except Exception as e:
            progress.empty()
            st.exception(e)

with st.expander("Install"):
    st.code(
        "pip install streamlit langchain langgraph langchain-openai openai pydantic\n"
        "streamlit run youtube_thumbnail_app.py",
        language="bash",
    )
