from icrawler.builtin import GoogleImageCrawler

attractionList = ['tower eiffel', 'statue of liberty', 'niagara falls']
for keyword in attractionList:
    google_crawler = GoogleImageCrawler(parser_threads=2, downloader_threads=4, storage={'root_dir': 'images/{}'.format(keyword)})
    google_crawler.crawl(keyword=keyword, max_num=5, min_size=(500,500))



