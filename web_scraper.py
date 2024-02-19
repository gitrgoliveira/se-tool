import asyncio
import logging
from asyncio import Semaphore
from multiprocessing import Manager
from sqlite3 import connect
from typing import List, Optional
from urllib.parse import urldefrag, urlparse

from bs4 import BeautifulSoup
from langchain.document_loaders.base import BaseLoader
# from langchain.schema.document import Document
from langchain.text_splitter import TextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents.base import Document
from langchain_core.utils.html import extract_sub_links
from playwright.async_api import Browser, Page, async_playwright


class Scraper(BaseLoader):
    manager = Manager()
    visited = manager.dict()  # Make this a shared dictionary
    semaphore = Semaphore(5)  # default to 5 concurrent tasks

    def __init__(self, base_url: str, max_depth: int = 0, concurrency: int = 5, prevent_outside = True, num_retries = 5, debug = False):
        self.base_url: str = self.normalize_url(base_url)
        self.max_depth: int = max_depth
        self.semaphore = Semaphore(concurrency)  # Limit concurrency to 5 tasks
        self.prevent_outside = prevent_outside
        self.num_retries = num_retries
        
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        if self.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)


    async def _metadata_extractor(self, page: Page, soup: BeautifulSoup, url: str) -> dict:
        metadata: dict = {"source": self.normalize_url(url)}
        metadata["title"] = await page.title()
        # metadata["title"] = await page.wait_for_load_state('domcontentloaded') or await page.evaluate("document.readyState") == 'complete' or True, (await page.title())

        if description := soup.find("meta", attrs={"name": "description"}):
            metadata["description"] = description.get("content", None)
        if html := soup.find("html"):
            metadata["language"] = html.get("lang", None)
        return metadata

    def load(self) -> List[Document]:
        return asyncio.run(self.async_load())

    async def async_load(self, text_splitter: Optional[TextSplitter] = None) -> List[Document]:
        documents: List[Document] = []
        browser = None
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True,chromium_sandbox=True, timeout=60000)
            try:
                await self._scrape_documents(self.base_url, 0, documents, browser, text_splitter)
            except Exception as e:
                self.logger.error(f"An error {type(e).__name__} occurred while scraping: {e}")
            finally:
                if browser:
                    await browser.close()
        return documents

    async def _scrape_documents(self, url: str, depth: int, documents: List[Document], browser: Browser, text_splitter: Optional[TextSplitter] = None):
        if not self.is_valid_url(url):
            self.logger.warning(f"Invalid URL: {url}")
            return
        url = self.normalize_url(url)
        if self.visited.get(url, False) or depth > self.max_depth:
            return
        self.visited[url] = True
        
        if url.endswith('.pdf'):
            try:
                self._handle_pdf(url, documents, text_splitter)
            except Exception as e:
                self.logger.error(f"An error {type(e).__name__} occurred while scraping {url}: {e}")
                
            return
        retry_count = 0
        while retry_count < self.num_retries:
            try:
                self.logger.debug(f"Added to queue {url}")
                async with self.semaphore:
                    # sleep_time = random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
                    # await asyncio.sleep(sleep_time) # sleeping for one second so we are not hammering the server
                    self.logger.debug(f"Loading {url}")
                    page = await browser.new_page(base_url=url, is_mobile=False, java_script_enabled=True, accept_downloads=False)
                    await page.goto(url)
                    # Scroll to the end of the page in a loop
                    await self.scroll_page(page)
                    
                    content = await page.content()
                    # now we can declare that we have properly visited the url
                    self.visited[url] = True
                    if content == None or content=="":
                        self.logger.error("page content returned empty", content)
                        if not page.is_closed():
                            await page.close()
                        break
                    else:
                        soup = BeautifulSoup(content, "html.parser")
                        document = Document(
                            page_content=content,
                            metadata=await self._metadata_extractor(page, soup, url),
                        )
                        if text_splitter != None:
                            self.logger.debug("Splitting documents from", url)
                            split_documents = text_splitter.split_documents([document])
                            documents.extend(split_documents)
                        else:
                            documents.append(document)

                    if not page.is_closed():
                        await page.close()
                    self.logger.debug(f"Finished {url}")
                
                if depth < self.max_depth:
                    await self._scrape_all_sublinks(url, depth, documents, browser, content)
                # Scrape all sublinks only if we do not exceed maximum depth 
                
                break
            except TimeoutError as e:
                retry_count += 1
                if retry_count < self.num_retries:
                    self.logger.warning(f"Timeout error while scraping {url}: {e}")
                    self.logger.warning(f"Retrying {url}...")
                    continue
                else: 
                    self.logger.error(f"Timeout error while scraping {url}: {e}")

                
                self.visited[url] = False
            except Exception as e:
                retry_count += 1
                if retry_count < self.num_retries:
                    self.logger.warning(f"An error {type(e).__name__} occurred while scraping {url}: {e}")
                    self.logger.warning(f"Retrying {url}...")
                    continue
                else:
                    self.logger.error(f"An error {type(e).__name__} occurred while scraping {url}: {e}")
                
                self.visited[url] = False
            finally:
                if not page.is_closed():
                    await page.close()

    def _handle_pdf(self, url, documents, text_splitter):
        loader = PyPDFLoader(url, extract_images=False)
        pages = loader.load()
        if text_splitter != None:
            split_documents = text_splitter.split_documents(pages)
            documents.extend(split_documents)
        else:
            documents.extend(pages)

    async def _scrape_all_sublinks(self, url, depth, documents, browser, content: str):
        links = extract_sub_links(raw_html=content, url=url, base_url=self.base_url, prevent_outside=self.prevent_outside)
        tasks = []
        for next_url in links:
            if self.prevent_outside and not next_url.startswith(self.base_url):
                continue
            
            if next_url not in self.visited:
                task = self._scrape_documents(next_url, depth + 1, documents, browser)
                tasks.append(task)
                        
        await asyncio.gather(*tasks)

    async def scroll_page(self, page):
        while True:
            await page.wait_for_load_state('domcontentloaded')
            scroll_height_before = await page.evaluate("document.body.scrollHeight")
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(10000)  # Wait for new content to load
            scroll_height_after = await page.evaluate("document.body.scrollHeight")
            if scroll_height_before == scroll_height_after:
                break
    @staticmethod
    def normalize_url(url: str) -> str:
        """Normalize a URL by removing the fragment and converting it to lowercase."""
        url, _ = urldefrag(url)  # Remove fragment
        url = url.lower()  # Convert to lowercase
        return url
            
    def load_and_split(
        self, text_splitter: Optional[TextSplitter] = None
    ) -> List[Document]:
        return asyncio.run(self.async_load(text_splitter))
    
    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Check if a URL is valid."""
        parsed = urlparse(url)
        return bool(parsed.scheme) and bool(parsed.netloc)