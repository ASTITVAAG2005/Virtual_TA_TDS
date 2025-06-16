import os
import json
import re
import requests
import time
from bs4 import BeautifulSoup
from pathlib import Path
from google import genai
from dotenv import load_dotenv
from PIL import Image

# === CONFIG ===
DISCOURSE_DIR = "Discourse_Content"
COURSE_DIR = "Course_Content"
OUTPUT_JSON = "data.json"
IMAGE_DIR = "downloaded_images"
TDS_BASE = "https://tds.s-anand.net/#/"
DISCOURSE_BASE = "https://discourse.onlinedegree.iitm.ac.in/t"

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
os.makedirs(IMAGE_DIR, exist_ok=True)

image_count = 0
start_time = time.time()

# === HELPERS ===
def slugify(text):
    text = re.sub(r'[^\w\s-]', '', text.lower())
    return re.sub(r'[\s_]+', '-', text).strip('-')

def extract_images_and_text(html):
    soup = BeautifulSoup(html, "html.parser")
    images = [img["src"] for img in soup.find_all("img") if "src" in img.attrs]
    text = soup.get_text(separator="\n").strip()
    return text, images

def describe_image(path, image_name):
    global image_count, start_time
    try:
        img = Image.open(path)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[img, "Describe this image in a concise and clear way that conveys all essential technical or educational content. Avoid visual style commentary or unnecessary elaboration."]
        )
        caption = response.text.strip()
        print(f"🖼️ Described {image_name}: {caption[:100]}{'...' if len(caption) > 100 else ''}")
        image_count += 1

        # Rate limit: pause after 13 images if under a minute
        if image_count % 13 == 0:
            elapsed = time.time() - start_time
            if elapsed < 60:
                wait_time = 60 - elapsed
                print(f"⏳ Waiting {wait_time:.1f} seconds to respect Gemini rate limit...")
                time.sleep(wait_time)
            start_time = time.time()

        return caption
    except Exception as e:
        print(f"⚠️ Failed to describe {path}: {e}")
        return "[Image description unavailable]"

def build_discourse_entries():
    entries = []
    for fname in os.listdir(DISCOURSE_DIR):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(DISCOURSE_DIR, fname), encoding="utf-8") as f:
            data = json.load(f)

        topic_id = data.get("id")
        title = data.get("title", "Untitled")
        slug = slugify(title)
        posts = data.get("post_stream", {}).get("posts", [])

        for post in posts:
            post_number = post.get("post_number")
            post_id = post.get("id")
            html = post.get("cooked", "")
            text, images = extract_images_and_text(html)
            image_descriptions = []

            idx = 0
            while True:
                found = False
                for ext in [".jpg", ".png", ".webp"]:
                    image_name = f"{topic_id}_{post_id}_{idx}{ext}"
                    image_path = os.path.join(IMAGE_DIR, image_name)
                    if os.path.exists(image_path):
                        caption = describe_image(image_path, image_name)
                        image_descriptions.append(f"[Image {idx+1}]: {caption}")
                        found = True
                        break
                if not found:
                    break
                idx += 1

            if image_descriptions:
                text += "\n\n" + "\n".join(image_descriptions)

            url = f"{DISCOURSE_BASE}/{slug}/{topic_id}/{post_number}"
            entries.append({
                "title": title,
                "source": url,
                "filename": fname,
                "content": text
            })
    return entries

def build_course_entries():
    entries = []
    for path in Path(COURSE_DIR).rglob("*.md"):
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        title = lines[0].replace("#", "").strip() if lines else path.stem
        content = "".join(lines)
        clean_name = path.name.replace(".md", "")
        source_url = f"{TDS_BASE}{clean_name}"

        entries.append({
            "title": title,
            "source": source_url,
            "filename": path.name,
            "content": content
        })
    return entries

# === MAIN ===
all_entries = build_discourse_entries() + build_course_entries()
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(all_entries, f, indent=2, ensure_ascii=False)

print(f"✅ Built {len(all_entries)} entries into {OUTPUT_JSON}")
