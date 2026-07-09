import requests
from bs4 import BeautifulSoup
import re


def scrape_url_text(url:str) -> str:
    """Go the website and pull down only the raw text"""

    try:
        headers = {"User-Agent":"Mozilla/5.0"}
        response = requests.get(url , headers=headers, timeout=10)
        response.raise_for_status()


        soup = BeautifulSoup(response.text , "html.parser")


        for garbage in soup(["script" , "style" , "nav" , "footer" , "header" , "aside"]):
            garbage.extract()

        paragraphs = soup.find_all('p')
        raw_text = " ".join([p.get_text(strip=True) for p in paragraphs])
        
        # Clean double spaces
        return re.sub(r'\s+', ' ', raw_text).strip()
    except Exception as e:
        print(f"Extraction Error at {url}: {e}")
        return ""