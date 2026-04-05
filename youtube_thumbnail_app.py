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
TRANSCRIPT_LIMIT = 12000


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


def get_image_client() -> OpenAI:
    """Return a raw OpenAI client for image generation."""


########################## Graph Node: Plan ############################
def plan_thumbnails(state: AppState):
    """Generate structured thumbnail concepts from a video transcript using an LLM."""


######################### Graph Node: Render ###########################
def render_thumbnails(state: AppState):
    """Generate actual thumbnail images for each planned concept via the OpenAI image API."""


######################### LangGraph Pipeline ###########################
def build_graph():
    """Compile a two-step LangGraph: plan concepts then render images."""


########################### Streamlit UI ################################
def main():  # sourcery skip: extract-method, use-named-expression
    pass


if __name__ == "__main__":
    main()
