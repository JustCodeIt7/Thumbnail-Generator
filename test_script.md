# 🎬 YouTube Video Transcript: AI YouTube Thumbnail Generator with LangGraph + OpenAI

---

**[INTRO — 0:00]**

Hey, what's up everyone — welcome back to the channel. So today we're building something that I think is genuinely useful if you make YouTube content, or honestly if you're just learning how to combine LLMs with agentic pipelines.

We're going to build an **AI-powered YouTube Thumbnail Generator** — and I don't just mean "ask ChatGPT to describe a thumbnail." I mean a full end-to-end app that takes a video transcript, reasons about what thumbnails would actually get clicks, and then _generates the images_ — all inside a clean Streamlit web UI.

Here's what we're using:

- **LangGraph** to orchestrate a two-node AI pipeline
- **LangChain + OpenAI** for structured concept generation
- **OpenAI's image generation API** to render the actual thumbnails
- And **Streamlit** to wrap it all in a usable interface

By the end of this video, you'll have a working app and a solid understanding of how to structure multi-step LLM workflows using LangGraph. Let's get into it.

---

**[OVERVIEW — 1:10]**

Before we touch any code, let me show you what the app actually does at a high level, because it'll make the code a lot easier to follow.

The flow is simple — three steps:

1. You paste a YouTube transcript into the UI and choose how many thumbnails you want.
2. A language model reads that transcript and generates structured thumbnail _concepts_ — things like a punchy headline, a visual scene description, a style direction, and a full image generation prompt.
3. Those concepts get passed to an image model, which renders each one as a real 16:9 PNG you can download.

The magic here is that steps two and three are wired together using **LangGraph** — a framework for building stateful, multi-step AI agent pipelines. Think of it like a directed graph where each node is an AI task and the state flows between them automatically.

Alright, let's walk through the code.

---

**[DATA MODELS — 2:30]**

Let's start at the top of the file, with the data models.

```python
TRANSCRIPT_LIMIT = 12000
```

First thing we do is cap the transcript at 12,000 characters. This is just a practical safeguard to avoid blowing through your token limit on the LLM call. Most transcripts will be way under this, but it's good defensive programming.

Next we define our Pydantic models:

```python
class ThumbnailIdea(BaseModel):
    headline: str = Field(description="Short overlay text, 2-5 words")
    hook: str = Field(description="Why this thumbnail is clickable")
    visual: str = Field(description="Main visual scene or composition")
    style: str = Field(description="Art direction and color mood")
    prompt: str = Field(description="Detailed image prompt for the thumbnail")
```

So `ThumbnailIdea` represents a single thumbnail concept. Each field maps to a specific dimension of what makes a good thumbnail — the **headline** is that big bold text you see in the thumbnail, the **hook** is the psychological reason someone would click it, the **visual** describes the imagery, the **style** describes the color mood and artistic direction, and the **prompt** is a full image generation prompt we can feed directly to DALL-E or `gpt-image-1`.

```python
class ThumbnailPlan(BaseModel):
    ideas: List[ThumbnailIdea] = Field(description="List of unique thumbnail concepts")
```

`ThumbnailPlan` just wraps a list of those ideas. This is the top-level object we ask the LLM to return.

Why Pydantic? Because we're using LangChain's **structured output** feature — we literally tell the model "return a JSON object that matches this schema," and LangChain handles the parsing and validation. No regex hacks, no fragile string parsing.

Then we have our state object:

```python
class AppState(TypedDict):
    transcript: str
    count: int
    ideas: list
    images: list
```

`AppState` is a TypedDict — this is LangGraph's way of defining what data flows through the pipeline. It carries the transcript and count in, accumulates ideas after the planning node, and then accumulates the rendered images after the render node. Every node reads from and writes back to this shared state.

---

**[MODEL INITIALIZATION — 5:00]**

```python
def get_text_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("TEXT_MODEL", "gpt-4.1-mini"),
        temperature=0.9,
    )
```

`get_text_model` returns a LangChain `ChatOpenAI` instance. We default to `gpt-4.1-mini`, but you can override it with an environment variable. The temperature is set to `0.9` — intentionally high, because we _want_ creative, diverse output here. If you set this to something like `0.2`, all your thumbnails would feel the same.

```python
def get_image_client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

For image generation, we're using the raw `openai` Python client rather than LangChain, because LangChain doesn't wrap the images API. We just grab the API key from the environment.

---

**[NODE 1: PLAN — 6:00]**

Now let's look at the first LangGraph node — the planning step.

```python
def plan_thumbnails(state: AppState):
    planner = get_text_model().with_structured_output(ThumbnailPlan, method="json_schema")
    transcript = state["transcript"][:TRANSCRIPT_LIMIT]
```

We take the text model and call `.with_structured_output(ThumbnailPlan, method="json_schema")`. This is a LangChain method that forces the model to return output matching our Pydantic schema using OpenAI's native JSON schema mode. The result is automatically deserialized into a `ThumbnailPlan` object — no extra parsing needed.

Then we write the prompt:

```python
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
```

Notice we inject the exact count into the prompt — "Create EXACTLY {count} unique thumbnail concepts." We also give it strong constraints: punchy text, one dominant subject, no stock-photo vibes, and diversity between ideas. This kind of structured prompt engineering is what separates decent AI output from genuinely useful output.

```python
    result = planner.invoke(prompt)
    ideas = [i.model_dump() for i in result.ideas[:state["count"]]]
    return {"ideas": ideas}
```

We invoke the planner, slice the results to enforce the count — just in case the model returned extras — and convert each `ThumbnailIdea` to a plain dictionary using `.model_dump()`. That's important because LangGraph state needs to be JSON-serializable. We return the ideas dict, and LangGraph merges it into the shared `AppState`.

---

**[NODE 2: RENDER — 8:30]**

The second node takes those planned concepts and turns them into actual images.

```python
def render_thumbnails(state: AppState):
    client = get_image_client()
    outputs = []
    used_heads = []
```

We initialize the OpenAI client, an empty list to accumulate outputs, and a `used_heads` list — this is a neat trick for keeping the generated images visually distinct from each other.

```python
    for idx, idea in enumerate(state["ideas"], start=1):
        diversity_note = "Previous headlines to avoid repeating: " + ", ".join(used_heads) if used_heads else ""
```

For every idea, we build a `diversity_note` string listing all previously used headlines. We inject this into the image prompt so the model knows not to repeat visual patterns. It's a simple but effective technique for batch diversity.

```python
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
```

The image prompt is structured and detailed. We're feeding in all five fields from our `ThumbnailIdea` plus hard constraints like "high contrast," "no watermark," "minimal clutter." The more specific your image prompt, the better the result — vague prompts produce vague images.

```python
        res = client.images.generate(
            model=os.getenv("IMAGE_MODEL", "gpt-image-1.5"),
            prompt=img_prompt,
            size="1536x1024",
            quality=os.getenv("IMAGE_QUALITY", "medium"),
            output_format="png",
        )
```

We call `client.images.generate` with `1536x1024` — that's a 16:9 landscape ratio, perfect for thumbnails. We request PNG output and default to `medium` quality, which balances speed and cost. You can bump this to `high` if you want production-quality images.

```python
        outputs.append({
            "index": idx,
            "headline": idea["headline"],
            "hook": idea["hook"],
            "visual": idea["visual"],
            "prompt": img_prompt,
            "b64": res.data[0].b64_json,
        })
        used_heads.append(idea["headline"])
    return {"images": outputs}
```

We append all the metadata plus the raw base64 PNG to our outputs list, then track the headline for diversity purposes on the next iteration. At the end, we return the images dict back into the pipeline state.

---

**[LANGGRAPH PIPELINE — 11:15]**

Now let's wire it all together.

```python
def build_graph():
    graph = StateGraph(AppState)
    graph.add_node("plan", plan_thumbnails)
    graph.add_node("render", render_thumbnails)
    graph.set_entry_point("plan")
    graph.add_edge("plan", "render")
    graph.add_edge("render", END)
    return graph.compile()
```

This is LangGraph in its simplest form. We create a `StateGraph` typed to our `AppState`, register two nodes — `plan` and `render` — set the entry point to `plan`, define a linear edge from `plan` to `render` and from `render` to `END`, and then compile it.

The compiled graph is a callable object. When you invoke it with an initial state, it runs through the nodes in order, passing the accumulated state between them automatically.

In a more complex app, you'd add conditional edges, loops, or parallel branches. But for this use case — plan, then render — a simple linear graph is exactly right. Don't over-engineer your agent topology.

---

**[STREAMLIT UI — 12:30]**

The UI is built with Streamlit, which is fantastic for AI prototypes because you write pure Python and get a web app with almost no boilerplate.

```python
st.set_page_config(page_title="AI Thumbnail Generator", page_icon="🎬", layout="wide")
st.title("🎬 AI YouTube Thumbnail Generator")
st.caption("Paste a transcript, choose a count, and generate unique thumbnail concepts + images.")
```

Standard Streamlit page config. Wide layout is important here because we're displaying 16:9 images — you want as much horizontal space as possible.

```python
with st.sidebar:
    count = st.number_input("How many thumbnails?", min_value=1, max_value=6, value=1, step=1)
    st.text_input("Text model", value=os.getenv("TEXT_MODEL", "gpt-4.1-mini"))
    st.text_input("Image model", value=os.getenv("IMAGE_MODEL", "gpt-image-1.5"))
    st.markdown("Set `OPENAI_API_KEY` before running the app.")
```

The sidebar holds the configuration inputs. We cap the count at 6 — mostly a cost guardrail, since each image generation call costs real money. The model fields are read-only displays of the current environment config — a nice way to surface what's running without making the UI editable.

```python
transcript = st.text_area(
    "YouTube video transcript",
    height=280,
    placeholder="Paste the full transcript here...",
)
generate = st.button("Generate thumbnails", type="primary", width="stretch")
```

The main body has a large text area for the transcript and a big primary button to kick off generation.

```python
if generate:
    graph = build_graph()
    progress = st.progress(0, text="Planning thumbnail concepts...")
    result = graph.invoke({
        "transcript": transcript.strip(),
        "count": int(count),
        "ideas": [],
        "images": [],
    })
    progress.progress(100, text="Done")
```

When the button is clicked, we build the graph, show a progress bar, and invoke the pipeline with the initial state. The `graph.invoke` call blocks until both nodes have completed — so you'll see the spinner while the LLM does its work and while images are being generated.

One thing to note: we initialize `ideas` and `images` as empty lists in the starting state. LangGraph will merge the return values from each node into this state as the graph executes, so by the time `invoke` returns, `result["images"]` will have all the generated thumbnails.

```python
    st.subheader("Concepts")
    for item in result["images"]:
        with st.expander(f"#{item['index']} — {item['headline']}", expanded=False):
            st.write(f"**Hook:** {item['hook']}")
            st.write(f"**Visual:** {item['visual']}")
            st.code(item["prompt"], language="text")
```

We display each concept in a collapsible expander showing the hook, visual description, and the full image prompt. This is useful for debugging and for understanding _why_ the model made the choices it did.

```python
    st.subheader("Generated thumbnails")
    cols = st.columns(2)
    for i, item in enumerate(result["images"]):
        img_bytes = base64.b64decode(item["b64"])
        with cols[i % 2]:
            st.image(img_bytes, caption=item["headline"], width="content")
            st.download_button(
                label=f"Download #{item['index']}",
                data=img_bytes,
                file_name=f"thumbnail_{item['index']}.png",
                mime="image/png",
                width="stretch",
            )
```

Finally, the generated images are laid out in a 2-column grid. We decode the base64 string back into raw bytes, display it with `st.image`, and add a download button right below each thumbnail. The `i % 2` trick cycles through the two columns evenly, no matter how many images we have.

---

**[SETUP & RUNNING — 16:00]**

To run this yourself, you need a few things set up. First, install the dependencies:

```bash
pip install streamlit langgraph langchain-openai openai pydantic rich python-dotenv
```

Then create a `.env` file in the same directory:

```
OPENAI_API_KEY=sk-...
TEXT_MODEL=gpt-4.1-mini
IMAGE_MODEL=gpt-image-1
IMAGE_QUALITY=medium
```

And run the app with:

```bash
streamlit run app.py
```

That's it. Paste in a transcript, hit the button, and watch your thumbnails generate.

A quick note on cost — image generation with `gpt-image-1` at medium quality costs around $0.04 to $0.07 per image depending on resolution and quality settings. So generating 4 thumbnails will run you somewhere in the $0.20 to $0.30 range. Not bad for a full production-quality thumbnail set.

---

**[WRAP UP — 17:15]**

Alright, let's recap what we built.

We created a two-node **LangGraph** pipeline — a **plan** node that uses a language model with structured output to generate detailed thumbnail concepts from a transcript, and a **render** node that takes those concepts and generates real images using OpenAI's image API.

We used **Pydantic** to enforce a strict output schema on the LLM, which makes the data reliable and easy to work with downstream. We used **LangGraph's AppState** to cleanly thread data through the pipeline. And we wrapped the whole thing in a **Streamlit** UI with a sidebar, progress indicator, expandable concept cards, a responsive image grid, and one-click downloads.

The patterns here — typed state, structured outputs, multi-step graph pipelines — are the same ones you'd use to build much more complex agents. Once you understand how LangGraph flows work, you can add loops, human-in-the-loop steps, conditional branching, parallel execution — the whole thing scales up nicely.

If you found this useful, drop a like and subscribe — I post new AI engineering tutorials every week. The full source code is linked in the description. Leave a comment if you have questions or if you build something cool with this.

Thanks for watching, and I'll see you in the next one. ✌️

---

_[END SCREEN — 18:00]_
