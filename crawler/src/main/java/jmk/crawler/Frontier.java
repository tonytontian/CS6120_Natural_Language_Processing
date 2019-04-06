package jmk.crawler;


import java.util.LinkedHashMap;
import java.util.NoSuchElementException;

public class Frontier {
	
	public boolean isEmpty() {
		return map.isEmpty();
	}
	
	public int size() {
		return map.size();
	}

	public boolean hasUrl(String url) {
		return map.containsKey(url);
	}
	
	public void add(int depth, String url) {
		if (!map.containsKey(url)) {
			map.put(url, new Entry(depth, url));
		}
	}
	
	public Entry removeOldest() {
		if (map.isEmpty()) {
			throw new NoSuchElementException("frontier is empty");
		} else {
			Entry e = map.values().iterator().next();
			map.remove(e.url);
			return e;
		}
	}
	
	public static class Entry {
		public int depth;
		public String url;
		
		public Entry(int depth, String url) {
			this.depth = depth;
			this.url = url;
		}
	}
	
	private LinkedHashMap<String, Entry> map = new LinkedHashMap<>();
}
