"""
Lecture Notes AI - MVP pipeline.

Run:
    python pipeline.py

Needs:
    input/<something>.mp4      (the Zoom recording - any .mp4 in input/)
    input/transcript.vtt       (the Zoom transcript)
    ANTHROPIC_API_KEY          (environment variable)

Produces:
    output/topics.json         (concepts + lecture summary returned by Gemini - for debugging)
    selected/concept_NN.jpg    (one chosen screenshot per concept)
    output/notes.pdf           (final notes: title + 2 screenshots per page,
                                  followed by a full lecture summary section)
"""

import os
import re
import json
import glob

import cv2
import numpy as np
from google import genai
from google.genai import types
from fpdf import FPDF

# ----------------------------------------------------------------------
# Paths / config
# ----------------------------------------------------------------------
INPUT_DIR = "input"
SELECTED_DIR = "selected"
OUTPUT_DIR = "output"
TOPICS_JSON = os.path.join(OUTPUT_DIR, "topics.json")
PDF_PATH = os.path.join(OUTPUT_DIR, "notes.pdf")

MODEL = "claude-opus-4-8"

# Frame search: look in the seconds *before* a concept ends (the board is most
# complete right before the slide changes), sampling this often.
SEARCH_WINDOW_SEC = 25.0
SAMPLE_EVERY_SEC = 1.0


# ----------------------------------------------------------------------
# 1. Transcript Timestamp Parsing
# ----------------------------------------------------------------------
def ts_to_sec(ts):
    """'00:06:21.000' or '00:06:21' -> seconds (float)."""
    ts = ts.replace(",", ".")
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)



def find_transcript():
    vtts = sorted(glob.glob(os.path.join(INPUT_DIR, "*.vtt")))
    if not vtts:
        raise FileNotFoundError(f"No .vtt found in {INPUT_DIR}/")
    return vtts[0]


# ----------------------------------------------------------------------
# 2 + 3. Ask Gemini for concept boundaries + a full lecture summary
# ----------------------------------------------------------------------
# Replace with your actual key or load from environment
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", "REMOVEDA"))

def get_concepts(vtt_text):
    """
    Single Gemini call that returns BOTH:
      - result["concepts"]        -> list of {title, start, end} for screenshot selection
      - result["lecture_summary"] -> full structured lecture notes/summary
    """

    prompt = f"""
You are preparing professional CAT coaching lecture notes.

The transcript below is from a recorded CAT/MBA preparation lecture.

Your task is to divide the lecture into teaching segments and generate professional lecture notes.

========================
IMPORTANT RULES
========================

IGNORE the following completely.

Do NOT create concepts for:

- Greetings
- Attendance
- Course introductions
- Motivation
- Revision of previous lectures
- Recaps
- Homework discussion
- Announcements
- Schedule discussion
- Administrative discussions
- Doubt sessions unrelated to new teaching
- General conversation

The first concept MUST begin only when today's teaching actually starts.

If the instructor is only talking without teaching new content, do NOT create a concept.

If a timestamp would likely correspond to a blank slide, title slide, Zoom screen, recap slide or talking-head explanation without written content, do NOT create a concept for it.

========================
CONCEPT SEGMENTATION RULES
========================

Do NOT identify only "major concepts."

Instead, SPLIT the lecture into teaching segments.

Create a NEW concept whenever ANY of the following occurs:

• A new CAT question begins.
• A new solved example begins.
• A new formula is introduced.
• A new shortcut or trick is introduced.
• A new theorem or property is explained.
• A different solving method begins.
• The instructor changes to a different slide containing different academic content.
• A completely different sub-topic begins.

DO NOT merge two different solved examples.

DO NOT merge two different CAT questions.

DO NOT merge two different techniques.

Err on the side of creating MORE concepts rather than fewer.

Every concept should correspond to ONE useful screenshot that a student would save as notes.

Concepts must be:

- chronological
- non-overlapping
- visually meaningful
- suitable for screenshot extraction

========================
OUTPUT
========================

For every concept return:

- title
- start timestamp
- end timestamp

Then summarize the ENTIRE lecture (NOT each concept).

Return ONLY valid JSON in EXACTLY this format:

{{
  "lecture_summary": {{
      "summary":"",

      "key_concepts":[
          ""
      ],

      "concept_details":[
          {{
              "title":"",
              "points":[
                  ""
              ]
          }}
      ],

      "problem_solving_strategy":[
          ""
      ],

      "common_mistakes":[
          ""
      ],

      "exam_tips":[
          ""
      ],

      "class_notes":""
  }},

  "concepts":[
      {{
          "title":"",
          "start":"00:00:00",
          "end":"00:00:00"
      }}
  ]
}}

========================
QUALITY REQUIREMENTS
========================

- Cover the complete teaching portion of the lecture.
- Do not leave large teaching gaps uncovered.
- Titles should be short (3-8 words).
- Timestamps should closely match when teaching of that concept begins and ends.
- Every concept should have enough teaching content to justify a screenshot.
- Do not invent concepts not present in the transcript.

The lecture notes should be concise, student-friendly and useful for CAT revision.

Transcript:

{vtt_text}
"""


    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
          temperature=0,
          top_p=0.1,
    ),
    )

    text = response.text.strip()

    # Gemini sometimes wraps JSON in ```json
    text = text.replace("```json", "").replace("```", "").strip()

    result = json.loads(text)

    return result


# ----------------------------------------------------------------------
# 4 + 5 + 6. Find the sharpest, most stable frame near a timestamp
# ----------------------------------------------------------------------
def sharpness(gray):
    """Laplacian variance: high = in focus / lots of detail (finished writing)."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def pick_frame(cap, fps, concept_start, concept_end):
    """
    Sample frames in [end - window, end] and return the BGR frame that best
    balances sharpness (detail) and stability (not mid-stroke).
    """
    window_start = max(concept_start, concept_end - SEARCH_WINDOW_SEC)

    times = []
    t = window_start
    while t <= concept_end:
        times.append(t)
        t += SAMPLE_EVERY_SEC

    best_frame = None
    best_score = -1.0
    prev_gray = None

    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Reject blank / almost blank slides
        white_ratio = np.mean(gray > 240)

        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges > 0)

        if white_ratio > 0.97 or edge_density < 0.003:
            prev_gray = gray
            continue

        if prev_gray is not None:
            diff = float(np.mean(cv2.absdiff(prev_gray, gray)))
            score = sharpness(gray) / (1.0 + diff)
            if score > best_score:
                best_score = score
                best_frame = frame
        prev_gray = gray

    return best_frame


# ----------------------------------------------------------------------
# 8. Build the PDF
# ----------------------------------------------------------------------
def safe(text):
    """fpdf2 core fonts are latin-1 only; replace anything outside it."""
    return text.encode("latin-1", "replace").decode("latin-1")


def section_header(pdf, title):
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(0, 10, safe(title), new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.ln(2)


def bullet_list(pdf, items):
    pdf.set_font("Helvetica", "", 11)

    for item in items:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 6, "- " + safe(str(item)))

    pdf.ln(2)

from fpdf import FPDF

class NotesPDF(FPDF):

    def header(self):
        # Watermark FIRST
        self.image("assets/watermark.png", x=45, y=95, w=120)

        # Header
        self.image("assets/header.png", x=0, y=0, w=210)

        self.set_y(25)

    def footer(self):
        self.set_y(-12)

        self.set_font("Helvetica", "", 9)

        self.cell(
            0,
            8,
            f"Page {self.page_no()}",
            align="C"
        )

def build_pdf(pages, summary, pdf_path, main_title):
    """
    pages   = [{'title', 'image'}, ...]
    summary = { ... }
    -> writes pdf_path
    """
    pdf = NotesPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ---- Main Title ----
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 12, safe(main_title), align="C")
    pdf.ln(8)

    # ---- Per-concept pages (2 images per page) ----
    count = 0
    for p in pages:
        if count == 2:
            pdf.add_page()
            count = 0

        pdf.set_font("Helvetica", "B", 12)
        pdf.multi_cell(0, 8, safe(p["title"]))
        pdf.ln(2)

        if p["image"] and os.path.exists(p["image"]):
            # Fit image to the page width (A4 usable width ~= 180mm).
            pdf.image(p["image"], w=180)
            pdf.ln(8) # Space before the next title/image

        count += 1

    # ---- Final lecture summary section ----
    if summary:
        pdf.add_page()
        
        pdf.set_font("Helvetica", "B", 18)
        pdf.multi_cell(0, 10, "SUMMARY")
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, safe(summary.get("summary", "")))
        pdf.ln(4)

        section_header(pdf, "KEY CONCEPTS")
        bullet_list(pdf, summary.get("key_concepts", []))

        concept_details = summary.get("concept_details", [])
        if concept_details:
            section_header(pdf, "CONCEPT DETAILS")
            for cd in concept_details:
                pdf.set_font("Helvetica", "B", 12)
                pdf.multi_cell(0, 7, safe(cd.get("title", "")))
                pdf.ln(1)
                bullet_list(pdf, cd.get("points", []))
                pdf.ln(2)

        section_header(pdf, "PROBLEM SOLVING STRATEGY")
        bullet_list(pdf, summary.get("problem_solving_strategy", []))

        section_header(pdf, "COMMON MISTAKES")
        bullet_list(pdf, summary.get("common_mistakes", []))

        section_header(pdf, "EXAM TIPS")
        bullet_list(pdf, summary.get("exam_tips", []))

        section_header(pdf, "CLASS NOTES")
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, safe(summary.get("class_notes", "")))

    pdf.output(pdf_path)


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def find_video():
    mp4s = sorted(glob.glob(os.path.join(INPUT_DIR, "*.mp4")))
    if not mp4s:
        raise FileNotFoundError(f"No .mp4 found in {INPUT_DIR}/")
    return mp4s[0]


def main():
    os.makedirs(SELECTED_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    video_path = find_video()
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    import re

    video_name = re.sub(r"\s*\(\d+\)$", "", video_name).strip()
    pdf_path = os.path.join(OUTPUT_DIR, f"{video_name}.pdf")
    print(f"Video:      {video_path}")
    vtt_path = find_transcript()

    print(f"Transcript: {vtt_path}")

    vtt_text = open(vtt_path, encoding="utf-8").read()

    print("Asking Gemini for concept boundaries + lecture summary...")
    result = get_concepts(vtt_text)
    concepts = result["concepts"]
    summary = result["lecture_summary"]

    json.dump(result, open(TOPICS_JSON, "w", encoding="utf-8"), indent=2)
    print(f"Gemini returned {len(concepts)} concepts + lecture summary -> {TOPICS_JSON}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    pages = []
    for i, c in enumerate(concepts, 1):
        start = ts_to_sec(c["start"])
        end = ts_to_sec(c["end"])
        print(f"[{i}/{len(concepts)}] {c['title']}  ({c['start']} - {c['end']})")

        frame = pick_frame(cap, fps, start, end)
        img_path = None
        if frame is not None:
            img_path = os.path.join(SELECTED_DIR, f"concept_{i:02d}.jpg")
            cv2.imwrite(img_path, frame)
        else:
            print("    (no frame found in window)")

        pages.append(
            {
                "title": f"{i}. {c['title']}",
                "image": img_path,
            }
        )

    cap.release()

    print("Building PDF...")
    # Pass the video_name to serve as the main title
    build_pdf(pages, summary, pdf_path, video_name)
    print(f"Done -> {pdf_path}")


if __name__ == "__main__":
    main()