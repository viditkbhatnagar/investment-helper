from typing import List, Dict, Any

import asyncio
import httpx
try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None  # type: ignore

try:
    from scrapy.crawler import CrawlerProcess  # type: ignore
    from scrapy import Spider  # type: ignore
except Exception:  # pragma: no cover
    CrawlerProcess = None  # type: ignore
    Spider = object  # type: ignore

try:
    from selenium import webdriver  # type: ignore
    from selenium.webdriver.chrome.options import Options  # type: ignore
    from selenium.webdriver.chrome.service import Service  # type: ignore
    from webdriver_manager.chrome import ChromeDriverManager  # type: ignore
except Exception:  # pragma: no cover
    webdriver = None  # type: ignore
    Options = None  # type: ignore
    Service = None  # type: ignore
    ChromeDriverManager = None  # type: ignore


async def fetch_html(url: str, timeout: int = 15) -> str:
    async with httpx.AsyncClient(timeout=timeout, headers={"user-agent": "investment-advisor/0.1"}) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.text


async def scrape_simple_news(urls: List[str]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    async def _one(url: str):
        try:
            html = await fetch_html(url)
            if not BeautifulSoup:
                return {"url": url, "title": title, "summary": ""}
            soup = BeautifulSoup(html, "lxml")  # type: ignore
            title = soup.title.string.strip() if soup.title and soup.title.string else url
            # naive meta description
            desc_tag = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
            desc = desc_tag.get("content").strip() if desc_tag and desc_tag.get("content") else ""
            return {"url": url, "title": title, "summary": desc}
        except Exception:
            return {"url": url, "title": url, "summary": ""}
    gathered = await asyncio.gather(*[_one(u) for u in urls], return_exceptions=True)
    for g in gathered:
        if isinstance(g, dict):
            results.append(g)
    return results


class GenericSpider(Spider):  # type: ignore
    name = "generic_spider"
    start_urls: List[str] = []  # type: ignore

    def parse(self, response):  # type: ignore
        title = response.css("title::text").get() or response.url
        desc = response.css('meta[name="description"]::attr(content)').get() or ""
        yield {"url": response.url, "title": title.strip(), "summary": (desc or "").strip()}


def crawl_with_scrapy(urls: List[str]) -> List[Dict[str, Any]]:
    if not CrawlerProcess or not isinstance(GenericSpider, type):
        return []
    items: List[Dict[str, Any]] = []
    class _CollectorPipeline:
        def process_item(self, item, spider):  # type: ignore
            items.append(dict(item))
            return item
    process = CrawlerProcess(settings={"LOG_ENABLED": False, "ITEM_PIPELINES": {__name__ + "._CollectorPipeline": 100}})  # type: ignore
    GenericSpider.start_urls = urls
    process.crawl(GenericSpider)
    process.start(stop_after_crawl=True)
    return items


def fetch_with_selenium(url: str) -> Dict[str, Any]:
    try:
        if not webdriver or not Options:
            return {"url": url, "title": url, "summary": ""}
        opts = Options()  # type: ignore
        opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        if ChromeDriverManager and Service:
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)  # type: ignore
        else:
            driver = webdriver.Chrome(options=opts)  # type: ignore
        driver.get(url)
        title = driver.title
        html = driver.page_source
        driver.quit()
        if not BeautifulSoup:
            return {"url": url, "title": title, "summary": ""}
        soup = BeautifulSoup(html, "lxml")  # type: ignore
        desc_tag = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
        desc = desc_tag.get("content").strip() if desc_tag and desc_tag.get("content") else ""
        return {"url": url, "title": title, "summary": desc}
    except Exception:
        return {"url": url, "title": url, "summary": ""}




