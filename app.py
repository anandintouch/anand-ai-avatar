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

import os
from pathlib import Path

import anthropic
import gradio as gr

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
    "Hi! I'm **Anand Prakash's AI Avatar**. Ask me anything about his 18+ years "
    "in IT — cloud-native applications, AI / GenAI & Agentic systems, his "
    "Executive Decision Intelligence platform, awards, certifications, and his "
    "strategic leadership in digital & cloud transformation.\n\n"
    "[Connect with Anand on LinkedIn](https://www.linkedin.com/in/anandprakash1/)"
)

demo = gr.ChatInterface(
    fn=respond,
    title="💼 Anand Prakash — AI Avatar",
    description=DESCRIPTION,
    examples=EXAMPLES,
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
