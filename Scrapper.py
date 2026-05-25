#loading libraries
import json
import re
from pathlib import Path
from typing import List, Optional
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import pandas as pd
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.http import Request, Response

# paths
BASE_PATH = Path("...")
BASE_PATH.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = BASE_PATH / "restaurant_reviews.xlsx"

# website URL

BASE = "https://www.speisekarte.de"
CITY_DIRECTORY_URLS = [
    f"{BASE}/staedteverzeichnis/{chr(c)}"
    for c in range(ord("a"), ord("z") + 1)
]

TEST_CITY_LIMIT = None
STARS_RE = re.compile(r"--stars-width:\s*([\d.]+)%", re.I)


def clean_space(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return re.sub(r"\s+", " ", str(value).strip()) or None


def clean_join(parts: List[str]) -> Optional[str]:
    values = [str(part).strip() for part in parts if part and str(part).strip()]
    if not values:
        return None
    return clean_space(" ".join(values))


def parse_style_rating(style: Optional[str]):
    if not style:
        return None, None

    match = STARS_RE.search(style)
    if not match:
        return None, None

    try:
        percent = float(match.group(1))
        return percent, percent / 20.0
    except Exception:
        return None, None


def city_name_from_path(path: str) -> str:
    slug = unquote(path.strip("/").split("/")[0])
    return slug.replace("-", " ").strip().title()


def parse_ld_json(response: Response) -> Optional[dict]:
    for raw in response.css("script[type='application/ld+json']::text").getall():
        raw = raw.strip()
        if not raw:
            continue

        try:
            data = json.loads(raw)
        except Exception:
            continue

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("@type") in ("Restaurant", "LocalBusiness"):
                    return item

        if isinstance(data, dict) and data.get("@type") in ("Restaurant", "LocalBusiness"):
            return data

    return None


def normalize_url_for_compare(url: str) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    page = query.get("page", ["1"])[0]
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}?page={page}"
# settings

class SpeisekarteFullSpider(scrapy.Spider):
    name = "speisekarte_full"

    custom_settings = {
        "ROBOTSTXT_OBEY": False,
        "LOG_LEVEL": "WARNING",
        "CONCURRENT_REQUESTS": 16,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 16,
        "DOWNLOAD_TIMEOUT": 20,
        "RETRY_TIMES": 3,
        "RETRY_HTTP_CODES": [429, 500, 502, 503, 504],
        "TELNETCONSOLE_ENABLED": False,
        "COOKIES_ENABLED": False,
        "DOWNLOAD_DELAY": 0.25,
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 0.5,
        "AUTOTHROTTLE_MAX_DELAY": 10.0,
        "DEFAULT_REQUEST_HEADERS": {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Upgrade-Insecure-Requests": "1",
        },
        "USER_AGENT": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.seen_city_directory_pages = set()
        self.discovered_city_paths = []
        self.discovered_city_set = set()
        self.city_requests_scheduled = False

        self.pending_directory_branches = len(CITY_DIRECTORY_URLS)

        self.seen_listing_pages = set()
        self.seen_restaurants = set()

    async def start(self):
        for url in CITY_DIRECTORY_URLS:
            yield Request(
                url=url,
                callback=self.parse_city_directory,
                dont_filter=True,
                meta={"directory_page_num": 1},
            )

    def start_requests(self):
        for url in CITY_DIRECTORY_URLS:
            yield Request(
                url=url,
                callback=self.parse_city_directory,
                dont_filter=True,
                meta={"directory_page_num": 1},
            )

    def parse_city_directory(self, response: Response):
        normalized_current = normalize_url_for_compare(response.url)

        if normalized_current in self.seen_city_directory_pages:
            return

        self.seen_city_directory_pages.add(normalized_current)

        new_count = 0

        for href in response.css("a[href$='/restaurants']::attr(href)").getall():
            href = (href or "").strip()

            if not href.startswith("/") or not href.endswith("/restaurants"):
                continue

            if href not in self.discovered_city_set:
                self.discovered_city_set.add(href)
                self.discovered_city_paths.append(href)
                new_count += 1

        next_url = self.find_next_directory_page(response) if new_count > 0 else None

        if next_url:
            normalized_next = normalize_url_for_compare(next_url)

            if normalized_next != normalized_current and normalized_next not in self.seen_city_directory_pages:
                yield Request(
                    url=next_url,
                    callback=self.parse_city_directory,
                    meta={
                        "directory_page_num": response.meta.get("directory_page_num", 1) + 1
                    },
                )
                return

        self.pending_directory_branches -= 1

        if self.pending_directory_branches == 0 and not self.city_requests_scheduled:
            self.city_requests_scheduled = True

            selected_city_paths = (
                self.discovered_city_paths
                if TEST_CITY_LIMIT is None
                else self.discovered_city_paths[:TEST_CITY_LIMIT]
            )

            for city_path in selected_city_paths:
                yield Request(
                    url=urljoin(BASE, city_path),
                    callback=self.parse_listing,
                    meta={"current_page_num": 1},
                )

    def find_next_directory_page(self, response: Response) -> Optional[str]:
        current_url = response.url

        candidates = [
            response.css("a[rel='next']::attr(href)").get(),
            response.xpath("//a[contains(normalize-space(.), 'Nächste')]/@href").get(),
            response.xpath("//a[contains(normalize-space(.), 'Weiter')]/@href").get(),
            response.xpath("//a[contains(normalize-space(.), 'Next')]/@href").get(),
        ]

        for href in candidates:
            if not href:
                continue

            next_url = urljoin(response.url, href)

            if normalize_url_for_compare(next_url) != normalize_url_for_compare(current_url):
                return next_url

        return None

    def parse_listing(self, response: Response):
        normalized_current = normalize_url_for_compare(response.url)

        if normalized_current in self.seen_listing_pages:
            return

        self.seen_listing_pages.add(normalized_current)

        for card in self.find_cards(response):
            rel = card.css("h2 a[href*='/restaurant/']::attr(href)").get()
            title = clean_space(card.css("h2 a::text").get())

            if not rel:
                continue

            restaurant_url = urljoin(response.url, rel)
            normalized_restaurant = restaurant_url.rstrip("/")

            if normalized_restaurant in self.seen_restaurants:
                continue

            self.seen_restaurants.add(normalized_restaurant)

            try:
                listing_city = city_name_from_path(urlparse(response.url).path)
            except Exception:
                listing_city = None

            yield Request(
                url=restaurant_url,
                callback=self.parse_detail,
                meta={
                    "restaurant_url": restaurant_url,
                    "restaurant_title_from_listing": title,
                    "listing_city": listing_city,
                    "page_url": response.url,
                },
            )

        next_url = self.find_next_page(response)

        if next_url:
            normalized_next = normalize_url_for_compare(next_url)

            if normalized_next != normalized_current and normalized_next not in self.seen_listing_pages:
                yield Request(
                    url=next_url,
                    callback=self.parse_listing,
                    meta={
                        "current_page_num": response.meta.get("current_page_num", 1) + 1
                    },
                )

    def find_cards(self, response: Response):
        cards = response.css("div.bg-white.shadow-md")
        valid_cards = []

        for card in cards:
            href = card.css("h2 a[href*='/restaurant/']::attr(href)").get()
            if href:
                valid_cards.append(card)

        if valid_cards:
            return valid_cards

        fallback_cards = []

        for link in response.css("h2 a[href*='/restaurant/']"):
            parent = link.xpath("ancestor::div[1]")
            if parent:
                fallback_cards.append(parent)

        return fallback_cards

    def find_next_page(self, response: Response) -> Optional[str]:
        current_url = response.url

        candidates = [
            response.css("a[rel='next']::attr(href)").get(),
            response.xpath("//a[contains(normalize-space(.), 'Nächste')]/@href").get(),
            response.xpath("//a[contains(normalize-space(.), 'Weiter')]/@href").get(),
            response.xpath("//a[contains(normalize-space(.), 'Next')]/@href").get(),
        ]

        for href in candidates:
            if not href:
                continue

            next_url = urljoin(response.url, href)

            if normalize_url_for_compare(next_url) != normalize_url_for_compare(current_url):
                return next_url

        return None

    def parse_detail(self, response: Response):
        restaurant_url = response.meta["restaurant_url"]

        restaurant_title = (
            self.extract_restaurant_title(response)
            or response.meta.get("restaurant_title_from_listing")
        )

        city = (
            self.extract_city(response, restaurant_url)
            or response.meta.get("listing_city")
        )

        bewertung_url = restaurant_url.rstrip("/") + "/bewertung"

        yield Request(
            url=bewertung_url,
            callback=self.parse_reviews,
            meta={
                "restaurant_url": restaurant_url,
                "bewertung_url": bewertung_url,
                "source_url": bewertung_url,
                "restaurant_title": restaurant_title,
                "city": city,
                "restaurant_address": self.extract_address(response),
                "detail_rating": self.extract_rating(response),
                "page_url": response.meta.get("page_url"),
            },
        )

    def extract_restaurant_title(self, response: Response) -> Optional[str]:
        title = clean_space(response.css("h1::text").get())
        if title:
            return title

        ld_data = parse_ld_json(response)
        if ld_data:
            return clean_space(ld_data.get("name"))

        return None

    def extract_city(self, response: Response, restaurant_url: str) -> Optional[str]:
        ld_data = parse_ld_json(response)

        if ld_data and isinstance(ld_data.get("address"), dict):
            city = clean_space(ld_data["address"].get("addressLocality"))
            if city:
                return city

        address = self.extract_address(response)

        if address:
            match = re.search(r"\b\d{5}\s+([A-Za-zÄÖÜäöüß\-\s]+)$", address)
            if match:
                return clean_space(match.group(1))

        try:
            return city_name_from_path(urlparse(restaurant_url).path)
        except Exception:
            return None

    def extract_address(self, response: Response) -> Optional[str]:
        address = clean_join(
            response.css("#detail-map p *::text").getall()
            or response.css("#detail-map p::text").getall()
        )

        if address:
            return address

        for label in ("Karte & Adresse", "Adresse"):
            heading = response.xpath(
                f"//*[self::h2 or self::h3 or self::p or self::div]"
                f"[translate(normalize-space(.), "
                f"'ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ', "
                f"'abcdefghijklmnopqrstuvwxyzäöü') = '{label.lower()}']"
            )

            if heading:
                text_parts = heading[0].xpath("following::p[1]//text()").getall()
                address = clean_join(text_parts)

                if address:
                    return address

        ld_data = parse_ld_json(response)

        if ld_data and isinstance(ld_data.get("address"), dict):
            address_data = ld_data["address"]
            parts = [
                clean_space(address_data.get("streetAddress")),
                clean_space(address_data.get("postalCode")),
                clean_space(address_data.get("addressLocality")),
            ]
            parts = [part for part in parts if part]

            if parts:
                return clean_space(", ".join(parts))

        return None

    def extract_rating(self, response: Response) -> Optional[str]:
        rating = clean_space(
            response.css("span.text-4xl.font-bold.text-speisekarte-red-100::text").get()
        )

        if rating:
            return rating.replace(",", ".")

        page_text = clean_join(response.xpath("//text()").getall()) or ""
        match = re.search(
            r"(\d+(?:[.,]\d+)?)\s*von\s*5\s*möglichen\s*Sternen",
            page_text,
            flags=re.I,
        )

        return match.group(1).replace(",", ".") if match else None

    def parse_reviews(self, response: Response):
        review_paragraphs = response.css("p.review-text")

        if not review_paragraphs:
            review_paragraphs = response.css("p[class*='review-text']")

        for review_paragraph in review_paragraphs:
            container = review_paragraph.xpath("ancestor::li[1]")

            if not container:
                container = review_paragraph.xpath("ancestor::article[1]")

            if not container:
                continue

            style = (
                container.css("span#rating-stars::attr(style)").get()
                or container.css("span.stars::attr(style)").get()
            )

            rating_percent, rating_stars = parse_style_rating(style)

            name_parts = (
                container.css("div.font-bold *::text").getall()
                or container.css("div.font-bold::text").getall()
                or container.css("div[class*='font-bold'] *::text").getall()
                or container.css("div[class*='font-bold']::text").getall()
            )

            reviewer_name = clean_join(name_parts)

            if reviewer_name and "|" in reviewer_name:
                reviewer_name = clean_space(reviewer_name.split("|", 1)[0])

            date_parts = (
                container.css("div.text-xs *::text").getall()
                or container.css("div.text-xs::text").getall()
                or container.css("div[class*='text-xs'] *::text").getall()
                or container.css("div[class*='text-xs']::text").getall()
            )

            review_text = clean_join(review_paragraph.xpath(".//text()").getall())

            if not review_text:
                continue

            # I keep the same columns as in the first scraping output.
            yield {
                "restaurant_url": response.meta["restaurant_url"],
                "bewertung_url": response.meta["bewertung_url"],
                "source_url": response.meta["source_url"],
                "restaurant_title": response.meta.get("restaurant_title"),
                "city": response.meta.get("city"),
                "restaurant_address": response.meta.get("restaurant_address"),
                "detail_rating": response.meta.get("detail_rating"),
                "page_url": response.meta.get("page_url"),
                "rating_percent": rating_percent,
                "rating_stars": rating_stars,
                "reviewer_name": reviewer_name,
                "review_date": clean_join(date_parts),
                "review_text": review_text,
            }


def main():
    scraped_items = []

    class CollectItemsPipeline:
        def process_item(self, item, spider):
            scraped_items.append(dict(item))
            return item

    process = CrawlerProcess(settings={
        "ITEM_PIPELINES": {CollectItemsPipeline: 100},
    })

    process.crawl(SpeisekarteFullSpider)
    process.start()

    columns = [
        "restaurant_url",
        "bewertung_url",
        "source_url",
        "restaurant_title",
        "city",
        "restaurant_address",
        "detail_rating",
        "page_url",
        "rating_percent",
        "rating_stars",
        "reviewer_name",
        "review_date",
        "review_text",
    ]

    df = pd.DataFrame(scraped_items, columns=columns)
    df.to_excel(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()