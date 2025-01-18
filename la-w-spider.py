import scrapy
import json
from scrapy.crawler import CrawlerProcess
from urllib.parse import urlparse


class LagenNuSpider(scrapy.Spider):
    name = 'lagen_nu'
    start_urls = ['https://lagen.nu/']
    custom_settings = {
        'DOWNLOAD_DELAY': 1,
        'ROBOTSTXT_OBEY': True
    }
    
    def __init__(self, *args, **kwargs):
        super(LagenNuSpider, self).__init__(*args, **kwargs)
        self.data = {"lagen": {"lagar": {}}}
        print("Spider initialized")

    def parse(self, response):
        print(f"Parsing page: {response.url}")
    
        # Get all links
        all_links = response.css('a::attr(href)').getall()
        print(f"Total number of links found: {len(all_links)}")
        print(f"First 10 links: {all_links[:10]}")

        base_domain = urlparse(response.url).netloc
    
        for href in all_links:
            parsed_href = urlparse(href)
            if (parsed_href.netloc == base_domain or not parsed_href.netloc) and not any(disallowed in href for disallowed in ['/api/', '/search/', '/-/']):
                full_url = response.urljoin(href)
                print(f"Following link: {full_url}")
                yield scrapy.Request(full_url, callback=self.parse_category)
            else:
                print(f"Skipping link: {href}")

        if not all_links:
            print("No links found on the page.")





    def parse_category(self, response):
        print(f"Parsing category: {response.url}")
        category = response.url.split('/')[-1]
        self.data["lagen"]["lagar"][category] = {}
        for href in response.css('article h2 a::attr(href)').extract():
            if not href.endswith('.png'):
                print(f"Following law link: {href}")
                yield response.follow(href, self.parse_law, meta={'category': category})

    def parse_law(self, response):
        print(f"Parsing law: {response.url}")
        category = response.meta['category']
        law_name = response.url.split('/')[-1]
        law = {
            "id": law_name,
            "title": response.css('h1::text').get(),
            "short_title": response.css('h1 small::text').get(),
            "metadata": {
                "department": response.css('.department::text').get(),
                "issued_date": response.css('.issued-date::text').get(),
                "last_amended": {"sfs": response.css('.last-amended::text').get()}
            },
            "chapters": []
        }
        
        print(f"Law title: {law['title']}")
        
        for chapter in response.css('section.chapter'):
            chapter_data = {
                "number": chapter.css('h2::text').get(),
                "title": chapter.css('h2 small::text').get(),
                "sections": []
            }
            print(f"Processing chapter: {chapter_data['number']}")
            
            for section in chapter.css('section.section'):
                section_data = {
                    "number": section.css('h3::text').get(),
                    "text": section.css('p::text').get(),
                    "comments": [{"text": comment, "type": "explanation"} for comment in section.css('.comment::text').getall()],
                    "amendments": [{"law": amendment} for amendment in section.css('.amendment::text').getall()],
                    "case_law": [{"count": len(section.css('.case-law-reference'))}],
                    "citations": [{"count": len(section.css('.citation-reference'))}],
                    "paragraphs": []
                }
                print(f"Processing section: {section_data['number']}")
                
                for paragraph in section.css('.paragraph'):
                    para_data = {
                        "text": paragraph.css('::text').get(),
                        "points": paragraph.css('li::text').getall()
                    }
                    section_data["paragraphs"].append(para_data)
                
                chapter_data["sections"].append(section_data)
            
            law["chapters"].append(chapter_data)
        
        self.data["lagen"]["lagar"][category][law_name] = law
        print(f"Finished parsing law: {law_name}")

    def closed(self, reason):
        print(f"Spider closed. Reason: {reason}")
        print(f"Saving data to lagen_nu_data.json")
        with open('lagen_nu_data.json', 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)
        print("Data saved successfully")

if __name__ == '__main__':
    print("Starting the spider")
    process = CrawlerProcess()
    process.crawl(LagenNuSpider)
    process.start()
    print("Spider process completed")
