# scrape_stories.py
import requests
from bs4 import BeautifulSoup
import re

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def scrape_stories(max_stories=50):
    base_url = "https://americanliterature.com/short-stories-for-children"
    stories = []

    r = requests.get(base_url)
    soup = BeautifulSoup(r.text, "html.parser")
    links = [a["href"] for a in soup.select("a") if a.get("href", "").startswith("/author")]

    for i, link in enumerate(links):
        if i >= max_stories:
            break
        url = "https://americanliterature.com" + link
        try:
            page = requests.get(url)
            s = BeautifulSoup(page.text, "html.parser")

            paragraphs = [p.get_text(" ", strip=True) for p in s.select("p")]
            story = clean_text(" ".join(paragraphs))

            if 300 < len(story.split()) < 800:
                stories.append(story)
                print(f"Scraped {i+1}: {url}")
        except Exception as e:
            print("Error:", e)
            continue

    return stories


if __name__ == "__main__":
    stories = scrape_stories(100)

    with open("stories_list.py", "w", encoding="utf-8") as f:
        f.write("stories = [\n")
        for s in stories:
            f.write(f'    """{s}""",\n')
        f.write("]\n")

    print(f"[+] Saved {len(stories)} stories into stories_list.py")
