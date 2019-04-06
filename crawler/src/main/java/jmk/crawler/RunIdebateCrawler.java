package jmk.crawler;

import java.net.URL;

public class RunIdebateCrawler {
	
	public static boolean shouldFollowUrl(URL url) {
		String s = url.toString();
		return s.startsWith("https://idebate.org/debatabase")
				&& !s.contains("rate=")
				&& !s.contains("user/login")
				&& !s.contains("offices-services");
	}

	public static void main(String[] args) {
		Crawler crawler = new Crawler(
				RunIdebateCrawler::shouldFollowUrl,
				(Crawler.LinkCallback) null);
		crawler.crawl("https://idebate.org/debatabase");
	}
}
