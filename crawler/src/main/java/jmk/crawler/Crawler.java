package jmk.crawler;


import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.BiPredicate;
import java.util.function.Predicate;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.jsoup.Connection;
import org.jsoup.Connection.Response;

import org.jsoup.HttpStatusException;
import org.jsoup.Jsoup;
import org.jsoup.UnsupportedMimeTypeException;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

/**
 * Crawler is a single-threaded web crawler that respects
 * robots.txt and may be configured to follow only links that
 * meet certain conditions.
 * 
 * This is essentially a single-use class: only one crawl per
 * instance of Crawler is likely to work.
 * 
 * @author Justin
 *
 */
public class Crawler {
	public static interface LinkCallback {
		void call(String currentUrl, String destUrl, String anchorText);
	}
	public static String USER_AGENT = "jmkcrawl";
	public static int POLITENESS_DELAY_MS = 5000;
	public static int MAX_VISITED_URLS = 10000;
	public static int MAX_DEPTH = 10;
	public static String DOCSTORE_PATH = "docstore";
	
	/**
	 * Creates a Crawler that pays no attention to link anchor text.
	 * 
	 * @param urlPredicate a test that returns true if the crawler should
	 *   follow the given URL
	 * @param urlListFileName the path to a file where the list of visited
	 *   URLs should be saved after a crawl
	 */
	public Crawler(Predicate<URL> urlPredicate, String urlListFileName) {
		this.linkPredicate = (u, t) -> urlPredicate.test(u);
		this.urlListFileName = urlListFileName;
	}

	/**
	 * Creates a Crawler that calls back when it encounters interesting links.
	 * 
	 * @param urlPredicate a test that returns true if the crawler should
	 *   follow the given URL
	 * @param linkCallback a function to be called on encountering an
	 *   interesting link (according to urlPredicate)
	 */
	public Crawler(Predicate<URL> urlPredicate, LinkCallback linkCallback) {
		this.linkPredicate = (u, t) -> urlPredicate.test(u);
		this.linkCallback = linkCallback;
	}

	/**
	 * Creates a Crawler that may focus on links that meet certain
	 * conditions.
	 * 
	 * @param linkPredicate a test that returns true if the crawler should
	 *   follow a link with the given URL and anchor text
	 * @param urlListFileName the path to a file where the list of visited
	 *   URLs should be saved after a crawl
	 */
	public Crawler(BiPredicate<URL, String> linkPredicate, String urlListFileName) {
		this.linkPredicate = linkPredicate;
		this.urlListFileName = urlListFileName;
	}
	
	public void crawl(String seedUrl) {
		int depth;
		frontier.add(1, seedUrl);
		
		do {
			Frontier.Entry next = frontier.removeOldest();
			String urlString = next.url;
			depth = next.depth;
			log.info("Removed from frontier: (" + depth + ") " + urlString);
			if (depth > MAX_DEPTH) {
				break;
			}
			
			if (!visitedUrls.contains(urlString)) {
				visitedUrls.add(urlString);
				
				// We'll either get this from the document store or download it
				Document doc;
				if (docStore.hasDocument(urlString)) {
					doc = loadDocumentFromStore(urlString);
				} else {
					doc = fetchDocumentFromUrl(urlString);
				}
				if (doc == null) {
					continue;
				}
				
				List<Link> links = getLinksInDocument(doc);
				for (Link link : links) {
					String s = link.destUrl.toString();
					if (isInterestingLink(link.destUrl, link.anchorText)) {
						if (linkCallback != null) {
							linkCallback.call(urlString, s, link.anchorText);
						}
						if (!visitedUrls.contains(s) && !frontier.hasUrl(s)) {
							frontier.add(depth + 1, s);
							log.finer("Added to frontier (" + (depth + 1) + ") " + s);
						}
					}
				}
			}
		} while (!frontier.isEmpty() && visitedUrls.size() < MAX_VISITED_URLS);
		
		log.info("Stopping at depth " + depth + " after visiting "
				+ visitedUrls.size() + " URLs");
		dumpUrlsToFile();
		docStore.close();
	}
	
	public void crawlDepthFirst(String seedUrl) {
		crawlDepthFirst(seedUrl, 1);
		
		log.info("Stopping at depth " + Math.min(MAX_DEPTH, visitedUrls.size())
				+ " after visiting " + visitedUrls.size() + " URLs");
		dumpUrlsToFile();
		docStore.close();
	}
	
	public Set<String> getVisitedUrls() {
		return Collections.unmodifiableSet(visitedUrls);
	}
	
	private void crawlDepthFirst(String urlString, int depth) {
		if (visitedUrls.contains(urlString) ||
				visitedUrls.size() >= MAX_VISITED_URLS) {
			return;
		}
		
		visitedUrls.add(urlString);
		log.info("Crawling depth first: (" + depth + ") " + urlString);
		
		// We'll either get this from the document store or download it
		Document doc;
		if (docStore.hasDocument(urlString)) {
			doc = loadDocumentFromStore(urlString);
		} else {
			doc = fetchDocumentFromUrl(urlString);
		}
		
		if (doc == null) {
			return;
		}
		
		List<Link> links = getLinksInDocument(doc);
		for (Link link : links) {
			if (isInterestingLink(link.destUrl, link.anchorText)) {
				String s = link.destUrl.toString();
				if (linkCallback != null) {
					linkCallback.call(urlString, s, link.anchorText);
				}
				if (depth < MAX_DEPTH &&
						!visitedUrls.contains(s) &&
						!frontier.hasUrl(s)) {
					crawlDepthFirst(link.destUrl.toString(), depth + 1);
				}
			}
		}
	}
	
	private static class Link {
		URL destUrl;
		String anchorText;
		
		public Link(URL destUrl, String anchorText) {
			this.destUrl = destUrl;
			this.anchorText = anchorText;
		}
	}
	
	private static Pattern charsetPattern = Pattern.compile("charset=([^\\s;]+)");
	
	private List<Link> getLinksInDocument(Document doc) {
		Elements linkElts = doc.select("a[href]");
		List<Link> links = new ArrayList<>(linkElts.size());
		for (Element link : linkElts) {
			String r = link.attr("abs:href");
			String text = link.html();  // inner html; usually just text
			try {
				// Remove the fragment, if it exists
				String s = r.indexOf('#') >= 0 ?
						r.substring(0, r.lastIndexOf('#')) : r;
				URL u = new URL(s);
				links.add(new Link(u, text));
			} catch (MalformedURLException e) {
				log.finer("Bad href: " + r);
			}
		}
		return links;
	}
	
	/** Returns null on failure */
	private Document loadDocumentFromStore(String url) {
		String charset = "UTF-8";
		String contentType = docStore.getDocumentContentType(url);
		Matcher m = charsetPattern.matcher(contentType);
		if (m.find()) {
			charset = m.group(1);
		}
		byte[] docBytes = docStore.getDocumentContent(url);
		InputStream in = new ByteArrayInputStream(docBytes);
		try {
			return Jsoup.parse(in, charset, url);
		} catch (IOException e) {
			log.warning("Could not parse stored document " + url);
			return null;
		}
	}
	
	/** Returns null on failure */
	private Document fetchDocumentFromUrl(String urlString) {
		URL url = null;
		try {
			url = new URL(urlString);
		} catch (MalformedURLException e) {
			// This should never happen, because the string should have
			// been parsed before.
			log.warning("Pulled bad URL from frontier: " + urlString);
			return null;
		}
		if (!isCrawlingAllowed(url)) {
			log.fine("Policy prohibits crawling this URL; continuing");
			return null;
		}
		
		delay();
		Connection conn = Jsoup.connect(urlString);

		try {
			Response resp = conn.execute();
			log.fine("Downloaded " + resp.header("Content-length") + " bytes");
			
			if (!urlString.equals(conn.request().url().toString())) {
				urlString = conn.request().url().toString();
				log.info("Redirected to " + urlString);
			}
			
			if (!docStore.hasDocument(urlString)) {
				docStore.saveDocument(urlString, resp.bodyAsBytes(), new Date(), resp.header("Content-Type"));
			}
			
			Document doc = resp.parse();
			return doc;
		} catch (UnsupportedMimeTypeException e) {
			log.fine("Unsupported MIME type; ignoring " + e.getMimeType());
		} catch (HttpStatusException e) {
			log.fine("Unsuccessful HTTP status " + e.getStatusCode());
		} catch (IOException e) {
			log.log(Level.WARNING, "Network error", e);
		}
		return null;
	}
	
	/* This really should be replaced by a per-domain rate limiter */
	private void delay() {
		try {
			Thread.sleep(POLITENESS_DELAY_MS);
		} catch (InterruptedException e) {
			// At the risk of being impolite, ...
		}
	}
	
	/**
	 * Determines whether we should add a given link to our frontier
	 * for eventual retrieval.
	 * 
	 * @param url the destination of the link
	 * @param anchorText the link text
	 * @return true if we might want to crawl the URL
	 */
	private boolean isInterestingLink(URL url, String anchorText) {
		return this.linkPredicate.test(url, anchorText);
	}
	
	/**
	 * Determines whether site policies permit crawling the given URL.
	 * This process may involve retrieving the robots.txt file for the
	 * website.
	 * 
	 * @param url the URL to be crawled
	 * @return true if we can crawl the URL
	 */
	private boolean isCrawlingAllowed(URL url) {
		int port = url.getPort() > 0 ? url.getPort() : url.getDefaultPort();
		String prefix = url.getProtocol() + "://" + url.getHost() + ":" + port;
		if (!robotsPolicies.containsKey(prefix)) {
			log.info("Attempting to download robots.txt for " + prefix);
			try {
				Connection conn = Jsoup.connect(prefix + "/robots.txt")
						.userAgent(USER_AGENT).ignoreContentType(true);
				String robotsText = conn.execute().body();
				robotsPolicies.put(prefix, new RobotsPolicy(robotsText, USER_AGENT));
				log.info("Parsed robots.txt");
			} catch (IOException e) {
				robotsPolicies.put(prefix, RobotsPolicy.MISSING_ROBOTS_TXT);
				log.info("Could not download/parse robots.txt due to " + e.toString());
			}
		}
		return !robotsPolicies.containsKey(prefix) ||
				robotsPolicies.get(prefix).allows(url.getPath());
	}
	
	private void dumpUrlsToFile() {
		if (urlListFileName == null) {
			return;
		}
		try (PrintStream ps = new PrintStream(new FileOutputStream(urlListFileName))) {
			ps.println(String.join(System.lineSeparator(), visitedUrls));
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}
	
	/** Function that determines whether we are interested in a link */
	private BiPredicate<URL, String> linkPredicate;
	
	/** Optional function to be called for each interesting link found */
	private LinkCallback linkCallback;
	
	/** Collection of URLs (with depth) to be visited */
	private Frontier frontier = new Frontier();

	/** Set of URLs for which documents have been retrieved */
	private Set<String> visitedUrls = new LinkedHashSet<>();
	
	/** Saved documents */
	private DocStore docStore = new DocStore(new File(DOCSTORE_PATH));
	
	/** Maps protocol/domain/port to robots.txt policy */
	private Map<String, RobotsPolicy> robotsPolicies = new HashMap<>();
	
	/** Name of file where we should dump a list of URLs visited */
	private String urlListFileName;
	
	private static Logger log = Logger.getLogger("jmk.crawler");
	static {
		log.setUseParentHandlers(false);
		ConsoleHandler h = new ConsoleHandler();
		h.setLevel(Level.INFO);
		log.addHandler(h);
		log.setLevel(Level.ALL);
	}
}
