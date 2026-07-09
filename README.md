---
title: Anand Prakash AI Avatar
emoji: 💼
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: true
license: mit
short_description: Ask about Anand's IT, AI/GenAI & leadership experience
---

# Anand Prakash — AI Avatar Agent

An interactive AI agent (powered by Claude) that answers visitor questions
about Anand Prakash's professional background: IT experience, skills,
achievements & awards, cloud-native applications, AI / GenAI & Agentic
systems, and strategic leadership in digital & cloud transformation.

## How it works

- `profile.md` — the knowledge base. **Edit this file** to update what the
  avatar knows (search for `[PLEASE UPDATE]` markers).
- `app.py` — Gradio chat UI + Claude API (streaming, prompt caching, guarded
  against off-topic questions and hallucinated facts).

## Run locally

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...   # from https://platform.claude.com
python app.py
# open http://localhost:7860
```

## Deploy to Hugging Face Spaces (recommended)

1. Create a free account at https://huggingface.co and click
   **New Space** → SDK: **Gradio** → CPU basic (free).
2. Upload `app.py`, `profile.md`, `requirements.txt`, and this `README.md`
   (the YAML header above configures the Space automatically). Or push via git:
   ```bash
   git init && git add . && git commit -m "AI Avatar"
   git remote add space https://huggingface.co/spaces/anandintouch/anand-ai-avatar
   git push space main
   ```
3. In the Space: **Settings → Variables and secrets → New secret** →
   name `ANTHROPIC_API_KEY`, value your API key. The Space restarts and
   goes live at `https://huggingface.co/spaces/anandintouch/anand-ai-avatar`.

## Add to LinkedIn Featured section

1. On LinkedIn: profile → **Add profile section** → **Recommended** →
   **Add featured** → **Add a link**.
2. Paste your Space URL and give it a title like
   *"💬 Chat with my AI Avatar — ask about my experience"* and a one-line
   description.
3. LinkedIn pulls the thumbnail automatically; visitors click the card and
   land directly in the chat.
