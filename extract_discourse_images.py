import os
import json
import re
import requests
from bs4 import BeautifulSoup

# === CONFIG ===
DISCOURSE_DIR = "Discourse_Content"
IMAGE_DIR = "downloaded_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# === HELPERS ===
def extract_images_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    return [img["src"] for img in soup.find_all("img") if "src" in img.attrs]

def download_image(url, save_path):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"✅ Downloaded: {save_path}")
        return True
    except Exception as e:
        print(f"⚠️ Failed: {url} → {e}")
        return False

def get_file_extension(url):
    ext = os.path.splitext(url)[1]
    return ext if ext else ".jpg"

# === MAIN ===
def extract_images():
    for fname in os.listdir(DISCOURSE_DIR):
        if not fname.endswith(".json"):
            continue

        with open(os.path.join(DISCOURSE_DIR, fname), encoding="utf-8") as f:
            data = json.load(f)

        topic_id = data.get("id")
        posts = data.get("post_stream", {}).get("posts", [])

        for post in posts:
            post_id = post.get("id")
            html = post.get("cooked", "")
            image_urls = extract_images_from_html(html)

            for i, url in enumerate(image_urls):
                ext = get_file_extension(url).split("?")[0]  # Clean URL parameters
                filename = f"{topic_id}_{post_id}_{i}{ext}"
                save_path = os.path.join(IMAGE_DIR, filename)
                download_image(url, save_path)

if __name__ == "__main__":
    extract_images()
