import requests
from bs4 import BeautifulSoup


def parse_site(url):
    response = requests.get(url, allow_redirects=True, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
    })

    final_url = response.url

    article_response = requests.get(final_url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
    })

    soup = BeautifulSoup(article_response.content, 'html.parser')
    headline = soup.find('h1')
    headline_text = headline.get_text(strip=True) if headline else 'No headline found'

    paragraphs = soup.select('.article__body p') or soup.find_all('p')
    article_body = '\n'.join(p.get_text(strip=True) for p in paragraphs)
    return article_body
