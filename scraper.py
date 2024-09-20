import requests
from bs4 import BeautifulSoup
from markdownify import markdownify
from urllib.parse import urljoin, urlparse, urldefrag
import os

def get_domain(url):
    return urlparse(url).netloc

def strip_fragment(url):
    return urldefrag(url)[0]

def crawl_and_convert(url, output_file):
    domain = get_domain(url)
    visited = set()
    scraped = set()

    def process_page(page_url):
        page_url = strip_fragment(page_url)

        if page_url in visited or get_domain(page_url) != domain:
            return

        visited.add(page_url)
        print(f"Scraping: {page_url}")

        if page_url in scraped:
            print(f"Already scraped: {page_url}. Skipping...")
            return

        response = requests.get(page_url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Replace images with their alt text
        for img in soup.find_all('img'):
            alt_text = img.get('alt', '')
            img.replace_with(f'[Image: {alt_text}]')

        # Convert to markdown
        markdown = markdownify(str(soup), heading_style="ATX")

        # Write to file
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"# {page_url}\n\n")
            f.write(markdown)
            f.write("\n\n")

        scraped.add(page_url)

        # Find and process links
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(page_url, href)
            if get_domain(full_url) == domain and full_url not in visited:
                process_page(full_url)

    # Start crawling from the initial URL
    process_page(url)

# Usage
url = "https://python.langchain.com/docs/introduction/"  # Replace with your target website
output_file = "output.md"

# Remove the output file if it already exists
if os.path.exists(output_file):
    os.remove(output_file)

crawl_and_convert(url, output_file)
print(f"Conversion complete. Output saved to {output_file}")