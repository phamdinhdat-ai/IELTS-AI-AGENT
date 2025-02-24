import scrapy
from scrapy.crawler import CrawlerProcess

# Create Spider class
class UneguiApartments(scrapy.Spider):
    name = 'unegui_apts'
    allowed_domains = ['www.unegui.mn']
    custom_settings = {'FEEDS': {'results1.csv': {'format': 'csv'}},
                       'USER_AGENT': "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/27.0.1453.93 Safari/537.36"}
    start_urls = [
        'https://www.unegui.mn/l-hdlh/l-hdlh-zarna/oron-suuts-zarna/'
    ]

    def parse(self, response):
        cards = response.xpath(
            '//li[contains(@class,"announcement-container")]')
        for card in cards:
            name = card.xpath(".//a[@itemprop='name']/@content").extract_first()
            price = card.xpath(".//*[@itemprop='price']/@content").extract_first()
            date = card.xpath("normalize-space(.//div[contains(@class,'announcement-block__date')]/text())").extract_first()
            city = card.xpath(".//*[@itemprop='areaServed']/@content").extract_first()

            yield {'name': name,
                   'price': price,
                   'city': city,
                   'date': date}

        next_url = response.xpath("//a[contains(@class,'red')]/parent::li/following-sibling::li/a/@href").extract_first()
        if next_url:
            # go to next page until no more pages
            yield response.follow(next_url, callback=self.parse)


    # main driver
if __name__ == "__main__":
    process = CrawlerProcess()
    process.crawl(UneguiApartments)
    process.start()