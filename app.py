"""AI Avatar Agent for Anand Prakash.

A public-facing chat agent that answers questions about Anand's IT
experience, skills, achievements, awards, cloud-native / AI / GenAI /
Agentic-systems work, and strategic leadership in digital and cloud
transformation. Knowledge comes from profile.md; the agent is instructed
not to invent facts beyond it.

Run locally:
    export ANTHROPIC_API_KEY=sk-ant-...
    pip install -r requirements.txt
    python app.py
"""

import base64
import os
from pathlib import Path

import anthropic
import gradio as gr

try:
    import spaces
except ImportError:
    # `spaces` only exists in the Hugging Face ZeroGPU runtime. Provide a
    # no-op fallback so the app also runs locally without that package.
    class _SpacesStub:
        def GPU(self, *args, **kwargs):
            if args and callable(args[0]):  # bare @spaces.GPU usage
                return args[0]
            return lambda fn: fn  # @spaces.GPU(...) usage

    spaces = _SpacesStub()


@spaces.GPU
def _zerogpu_warmup():
    # No-op that satisfies ZeroGPU's requirement for a @spaces.GPU function
    # at startup. This app does all inference via the Anthropic API, so no
    # GPU is ever used; the chat itself runs on the Space's CPU.
    return None

MODEL = "claude-opus-4-8"
MAX_TOKENS = 2048
MAX_HISTORY_TURNS = 20  # cap context sent per request

PROFILE = Path(__file__).with_name("profile.md").read_text(encoding="utf-8")

SYSTEM_PROMPT = f"""You are the AI Avatar of Anand Prakash, speaking on his \
behalf to visitors from his LinkedIn profile. You answer questions about his \
professional background: IT experience, skills, achievements and awards, \
cloud-native application work, AI / GenAI / Agentic-systems expertise, his \
Executive Decision Intelligence platform venture, and his strategic \
leadership in digital and cloud transformation.

Rules:
- Answer ONLY from the profile below. If asked something not covered, say \
you don't have that detail and suggest connecting with Anand directly on \
LinkedIn (https://www.linkedin.com/in/anandprakash1/).
- Never invent roles, dates, employers, metrics, or credentials.
- Speak in third person about Anand ("Anand led...", "His experience \
includes...") in a warm, professional, recruiter-friendly tone.
- Keep answers concise (2-6 sentences or a short bullet list) unless the \
visitor asks for depth.
- Politely decline off-topic requests (coding help, opinions on other \
people, personal/private matters) and steer back to Anand's background.
- If a visitor wants to hire, partner, or invest, encourage them to reach \
out via LinkedIn.

=== ANAND PRAKASH PROFILE ===
{PROFILE}
=== END PROFILE ==="""

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment


def respond(message, history):
    messages = [
        {"role": turn["role"], "content": turn["content"]}
        for turn in history[-MAX_HISTORY_TURNS:]
        if turn["role"] in ("user", "assistant") and isinstance(turn["content"], str)
    ]
    messages.append({"role": "user", "content": message})

    try:
        with client.messages.stream(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=messages,
        ) as stream:
            partial = ""
            for text in stream.text_stream:
                partial += text
                yield partial
    except anthropic.RateLimitError:
        yield "I'm getting a lot of questions right now — please try again in a minute."
    except anthropic.APIStatusError as e:
        yield f"Sorry, something went wrong on my side (error {e.status_code}). Please try again."
    except anthropic.APIConnectionError:
        yield "I couldn't reach my brain just now — please check back shortly."


EXAMPLES = [
    "What is Anand's experience with AI, GenAI and Agentic systems?",
    "Tell me about the Executive Decision Intelligence platform he is building.",
    "What awards and recognition has Anand received?",
    "What is his experience in digital and cloud transformation leadership?",
    "What cloud-native technologies has he worked with?",
    "What certifications does Anand hold?",
]

DESCRIPTION = (
    "Hi! I'm **Anand Prakash's AI Avatar**. Ask me anything about his 20+ years "
    "in IT — Cloud-native applications, AI / GenAI & Agentic systems, his "
    "Executive Decision Intelligence platform, awards, certifications, and his "
    "strategic leadership in digital & cloud transformation.\n\n"
    "[Connect with Anand on LinkedIn](https://www.linkedin.com/in/anandprakash1/)"
)

TITLE = "Anand Prakash — AI Avatar"


def _avatar_img_tag():
    """Return an <img> tag for Anand's photo (avatar.jpg/.png/.webp next to
    this script) as a base64 data URI, or "" if no such file exists. Base64
    means it paints immediately as raw HTML — no file serving needed."""
    mimes = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
             ".png": "image/png", ".webp": "image/webp"}
    for suffix, mime in mimes.items():
        path = Path(__file__).with_name("avatar" + suffix)
        if path.exists():
            b64 = base64.b64encode(path.read_bytes()).decode()
            return (
                f'<img src="data:{mime};base64,{b64}" alt="Anand Prakash" '
                'style="width:72px;height:72px;border-radius:50%;flex:0 0 auto;'
                'object-fit:cover;border:3px solid #4f46e5;'
                'box-shadow:0 2px 8px rgba(0,0,0,0.15)">'
            )
    return ""


# Header: avatar (if present) to the left of the title text, centered as a row.
HEADER_HTML = (
    '<div style="display:flex;align-items:center;justify-content:center;'
    'gap:16px;margin:0.75rem 0 0.25rem;flex-wrap:wrap">'
    + _avatar_img_tag()
    + f'<h1 style="margin:0;font-size:1.9rem;font-weight:700">💼 {TITLE}</h1>'
    + '</div>'
)

with gr.Blocks(title=TITLE) as demo:
    gr.HTML(HEADER_HTML)
    gr.ChatInterface(
        fn=respond,
        description=DESCRIPTION,
        examples=EXAMPLES,
    )

if __name__ == "__main__":
    # Use PORT if explicitly set; otherwise let Gradio auto-pick a free port
    # (scanning upward from 7860) so a busy port doesn't crash startup.
    port = os.environ.get("PORT")
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(port) if port else None,
    )
