import os
import base64
from typing import List, TypedDict

import streamlit as st
from pydantic import BaseModel, Field
from openai import OpenAI
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from rich import print
from dotenv import load_dotenv

load_dotenv()


############################# Data Models #############################
# Cap transcript length to stay within token limits
TRANSCRIPT_LIMIT = 30_000


# Define structured output schema for a single thumbnail concept
class ThumbnailIdea(BaseModel):
    """A single thumbnail concept with structured components. Designed for parsing LLM output into a consistent format."""

    headline: str = Field(
        description="Short overlay text, 2-5 words"
    )  # headline is critical for CTR, should be punchy and clear
    hook: str = Field(
        description="Why this thumbnail is clickable"
    )  # hook explains the click appeal in a concise way, guiding the visual direction
    visual: str = Field(
        description="Main visual scene or composition"
    )  # visual describes the dominant imagery or composition
    style: str = Field(
        description="Art direction and color mood"
    )  # style describes the overall artistic direction and color palette
    prompt: str = Field(
        description="Detailed image prompt for the thumbnail"
    )  # prompt provides a detailed description for image generation


# Wrap multiple ideas into one response object for structured parsing
class ThumbnailPlan(BaseModel):
    """A collection of unique thumbnail concepts. Designed for parsing LLM output into a consistent format."""

    ideas: List[ThumbnailIdea] = Field(description="List of unique thumbnail concepts")


# Track data flowing through the LangGraph pipeline
class AppState(TypedDict):
    """The state of the application at a given point in the pipeline."""

    transcript: str
    count: int  # Number of thumbnails to generate
    ideas: list  # List of thumbnail concepts (dicts) generated from the plan node
    images: list  # List of generated images with metadata from the render node


######################### Model Initialization #########################
def get_text_model() -> ChatOllama:
    """Return a ChatOllama instance configured for creative thumbnail ideation."""
    return ChatOllama(
        model=os.getenv("TEXT_MODEL", "gemma4:e2b"),
        temperature=0.9,  # High temperature for diverse creative output
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
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
    # Limit transcript length to stay within token limits
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

    # Generate the structured output and parse it into our ThumbnailPlan model
    result = planner.invoke(prompt)
    # Enforce the requested count and convert to plain dicts for state serialization
    ideas = [i.model_dump() for i in result.ideas[: state["count"]]]
    # Add the ideas to the state for the next node
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

        # Generate the image and store the base64 result along with metadata for display and download
        res = client.images.generate(
            model=os.getenv("IMAGE_MODEL", "gpt-image-1.5"),
            prompt=img_prompt,
            size="1536x1024",  # 16:9 landscape ratio
            quality=os.getenv("IMAGE_QUALITY", "low"),
            output_format="png",
        )
        # Append the result with all relevant metadata for later display and download in the UI
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
    # Define the graph and add nodes for planning and rendering, with edges to define the flow
    graph = StateGraph(AppState)
    # The plan node generates structured thumbnail concepts and updates the state with 'ideas'
    graph.add_node("plan", plan_thumbnails)
    # The render node takes the planned ideas and generates images, updating the state with 'images'
    graph.add_node("render", render_thumbnails)

    graph.set_entry_point("plan")

    # Define the edges to control the flow of execution: plan → render → END
    graph.add_edge("plan", "render")
    graph.add_edge("render", END)
    return graph.compile()


########################### Streamlit UI ################################
def main():
    st.set_page_config(
        page_title="AI Thumbnail Generator", page_icon="🎬", layout="wide"
    )
    st.title("🎬 AI YouTube Thumbnail Generator")
    st.caption(
        "Paste a transcript, choose a count, and generate unique thumbnail concepts + images."
    )

    # Render sidebar with configuration inputs
    with st.sidebar:
        st.header("Settings")
        # Allow user to select how many thumbnail concepts to generate, with a reasonable range
        count = st.number_input(
            "How many thumbnails?", min_value=1, max_value=6, value=1, step=1
        )
        # Display the current model settings, allowing overrides via environment variables for flexibility
        st.text_input("Text model", value=os.getenv("TEXT_MODEL", "gemma4:e2b"))
        st.text_input("Image model", value=os.getenv("IMAGE_MODEL", "gpt-image-1.5"))
        st.markdown("Set `OPENAI_API_KEY` before running the app.")
    # Main input area for the YouTube video transcript, with a large text area for easy pasting and editing
    transcript = st.text_area(
        "YouTube video transcript",
        height=280,
        placeholder="Paste the full transcript here...",
    )

    ######################### Generation Handler ############################
    # Button to trigger the thumbnail generation process, which will run the LangGraph pipeline when clicked
    generate = st.button("Generate thumbnails", type="primary", width="stretch")

    # Validate inputs then run the full plan → render pipeline
    if generate:
        graph = build_graph()
        progress = st.progress(0, text="Planning thumbnail concepts...")
        # Run the graph with initial state and update progress as we go
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
        # Loop through the generated ideas and display their components
        for item in result["images"]:
            # Create an expander for each thumbnail concept, showing its index and headline i
            with st.expander(f"#{item['index']} — {item['headline']}", expanded=False):
                # Display the key components of the thumbnail concept in a clear format,
                st.write(f"**Hook:** {item['hook']}")
                st.write(f"**Visual:** {item['visual']}")
                st.code(item["prompt"], language="text")

        # Render thumbnails in a 2-column grid with download buttons
        st.subheader("Generated thumbnails")
        cols = st.columns(2)  # Use 2 columns for better visibility of thumbnails
        for i, item in enumerate(result["images"]):
            # Decode the base64 image data and display it in the appropriate column
            img_bytes = base64.b64decode(item["b64"])
            # Cycle through columns for even distribution
            with cols[i % 2]:
                # Display the thumbnail image with a caption of the headline, and provide a download button for each image with a clear label and appropriate file naming
                st.image(img_bytes, caption=item["headline"], width="content")
                st.download_button(
                    label=f"Download #{item['index']}",
                    data=img_bytes,
                    file_name=f"thumbnail_{item['index']}.png",
                    mime="image/png",
                    width="stretch",
                )


if __name__ == "__main__":
    main()
