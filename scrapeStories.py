
import requests
from bs4 import BeautifulSoup
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'([.,!?"\';:])', r' \1 ', text)
    text = re.sub(r'[^a-z0-9\s.,!?"\';:]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def scrape_stories(max_stories=500):
    base_url = "https://americanliterature.com/short-stories-for-children"
    stories = []

    for page_num in range(1, 50):
        if len(stories) >= max_stories:
            break
        url = f"{base_url}?page={page_num}"
        print(f"Scraping page: {url}")
        try:
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                print(f"Failed to fetch page {page_num}, status code: {r.status_code}")
                continue
        except requests.RequestException as e:
            print(f"Error fetching page {page_num}: {e}")
            continue

        soup = BeautifulSoup(r.text, "html.parser")
        links = [a["href"] for a in soup.select("a") if "/short-story/" in a.get("href", "")]

        for href in links:
            if len(stories) >= max_stories:
                return stories

            story_url = "https://americanliterature.com" + href
            try:
                page = requests.get(story_url, timeout=10)
                if not page.ok:
                    continue
                s = BeautifulSoup(page.text, "html.parser")
                paragraphs = [p.get_text(" ", strip=True) for p in s.select('article p')]
                if not paragraphs:
                    paragraphs = [p.get_text(" ", strip=True) for p in s.select('p')]
                story = clean_text(" ".join(paragraphs))
                word_count = len(story.split())
                if 200 < word_count < 2000:
                    stories.append(story)
                    print(f"[{len(stories)}/{max_stories}] {story_url} ({word_count} words)")
            except Exception as e:
                print(f"Error processing story {story_url}: {e}")
                continue
    return stories

if __name__ == "__main__":
    stories_data = scrape_stories(500)
    with open("stories_list.py", "w", encoding="utf-8") as f:
        f.write("stories = [\n")
        for s in stories_data:
            s_escaped = s.replace('\\', '\\\\').replace('"""', '\\"\\"\\"')
            f.write(f'    """{s_escaped}""",\n')
        f.write("]\n")
    print(f"\n[+] Saved {len(stories_data)} stories into stories_list.py")