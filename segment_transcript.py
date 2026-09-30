from pathlib import Path
import anthropic

# ==========================
# CLAUDE API KEY
# ==========================
client = anthropic.Anthropic(
    api_key="YOUR_CLAUDE_API_KEY"
)

# ==========================
# READ VTT FILE
# ==========================
transcript = Path(
    "input/transcript.vtt"
).read_text(
    encoding="utf-8"
)

# ==========================
# PROMPT
# ==========================
prompt = f"""
You are analyzing a CAT preparation lecture transcript.

The transcript is in VTT format and contains timestamps.

Your task:

1. Identify the major concepts/topics discussed.
2. Group related discussion into a single topic.
3. Ignore greetings, attendance, jokes, and administrative discussion.
4. Create meaningful study-note sections.
5. Use timestamps from the transcript.

Return ONLY valid JSON.

Example:

[
  {{
    "title": "Introduction to Graphs",
    "start_time": "00:02:15",
    "end_time": "00:08:30"
  }},
  {{
    "title": "Linear Graphs",
    "start_time": "00:08:30",
    "end_time": "00:18:10"
  }}
]

Transcript:

{transcript}
"""

# ==========================
# CLAUDE CALL
# ==========================
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=4000,
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

# ==========================
# SAVE RESULT
# ==========================
result = response.content[0].text

Path("output/topics.json").write_text(
    result,
    encoding="utf-8"
)

print("Topics saved to output/topics.json")