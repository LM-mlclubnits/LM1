# scrape_stories.py
import requests
from bs4 import BeautifulSoup
import re

def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

def scrape_stories(max_stories=200):
    base_url = "https://americanliterature.com/short-stories-for-children"
    stories = []

    for page_num in range(1, 30):  # pagination support
        url = f"{base_url}?page={page_num}"
        r = requests.get(url)
        if r.status_code != 200:
            break

        soup = BeautifulSoup(r.text, "html.parser")
        links = [a["href"] for a in soup.select("a") if "/short-story/" in a.get("href", "")]

        for href in links:
            if len(stories) >= max_stories:
                return stories

            story_url = "https://americanliterature.com" + href
            try:
                page = requests.get(story_url)
                if not page.ok:
                    continue
                s = BeautifulSoup(page.text, "html.parser")

                paragraphs = [p.get_text(" ", strip=True) for p in s.select("p")]
                story = clean_text(" ".join(paragraphs))

                word_count = len(story.split())
                if 200 < word_count < 2000:
                    stories.append(story)
                    print(f"[{len(stories)}] {story_url} ({word_count} words)")
            except Exception as e:
                print("Error:", e)
                continue

    return stories


if __name__ == "__main__":
    stories = scrape_stories(500)

    with open("stories_list.py", "w", encoding="utf-8") as f:
        f.write("stories = [\n")
        for s in stories:
            f.write(f'    """{s}""",\n')
        f.write("]\n")

    print(f"[+] Saved {len(stories)} stories into stories_list.py")
