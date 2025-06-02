import os
from pathlib import Path

from finvizfinance.news import News
from tqdm import tqdm

from deli import parse_site

class Scraper:
    def __init__(self):
        self.fnews = News()
        if not Path.is_dir(Path("data/scraped_news")):
            os.mkdir("data/scraped_news")

    def scrape_and_save(self):
        all_news = self.fnews.get_news()
        for i in tqdm(range(len(all_news["news"]))):
            text = parse_site(all_news["news"].iloc[i]["Link"])
            if len(text) > 600:
                with open(f"data/scraped_news/{hash(text)}.txt", "w") as f:
                    f.write(text)
