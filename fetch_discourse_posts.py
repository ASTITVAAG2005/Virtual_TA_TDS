import requests
import os
import json
import time
from datetime import datetime, timezone

# --- Configuration ---
DISCOURSE_BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"
CATEGORY_ID = 34  # TDS KB category
START_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2025, 4, 14, tzinfo=timezone.utc)
OUTPUT_DIR = "Discourse_Content"
PAGES_TO_FETCH = 40  # Adjust as needed

# --- Load cookie from file ---
COOKIE_FILE = "cookies.txt"
if not os.path.exists(COOKIE_FILE):
    raise FileNotFoundError("cookies.txt not found. Please save your Discourse session cookie there.")

with open(COOKIE_FILE, "r") as f:
    cookie = f.read().strip()

HEADERS = {
    "Cookie": cookie,
    "User-Agent": "Mozilla/5.0"
}

# --- Setup ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Fetch topic list page-by-page ---
def fetch_topics(page):
    url = f"{DISCOURSE_BASE_URL}/c/courses/tds-kb/{CATEGORY_ID}.json?page={page}"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.json().get("topic_list", {}).get("topics", [])

# --- Fetch full topic (including posts) ---
def fetch_topic(topic_id):
    url = f"{DISCOURSE_BASE_URL}/t/{topic_id}.json"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.json()

# --- Main execution ---
def main():
    seen_ids = set()
    for page in range(PAGES_TO_FETCH):
        print(f"Fetching topic list page {page+1}...")
        try:
            topics = fetch_topics(page)
        except Exception as e:
            print("Error fetching topic list:", e)
            break

        for topic in topics:
            try:
                topic_id = topic["id"]
                created_at = datetime.fromisoformat(topic["created_at"].replace("Z", "+00:00"))

                if not (START_DATE <= created_at <= END_DATE):
                    continue
                if topic_id in seen_ids:
                    continue

                print(f"  Downloading topic ID {topic_id}: {topic['title'][:60]}")
                topic_json = fetch_topic(topic_id)
                filename = os.path.join(OUTPUT_DIR, f"{topic_id}.json")
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(topic_json, f, indent=2)

                seen_ids.add(topic_id)
                time.sleep(1)  # Be nice to the server
            except Exception as e:
                print("  Error fetching topic:", e)
                continue

if __name__ == "__main__":
    main()
