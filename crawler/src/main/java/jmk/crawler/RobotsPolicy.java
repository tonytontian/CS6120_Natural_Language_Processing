package jmk.crawler;


import java.util.ArrayList;
import java.util.List;

/**
 *  A RobotsPolicy is used to determine whether our crawler may access
 *  a given path on a site, according to the wishes of the site
 *  administrators, as expressed in a robots.txt file.
 */
public class RobotsPolicy {
	
	/** The policy to use when a site does not have a robots.txt */
	public static RobotsPolicy MISSING_ROBOTS_TXT =
			new RobotsPolicy("User-agent: *\nDisallow: \n", "*");

	/**
	 * Creates a policy from the given robots.txt, applied to the
	 * given user-agent.
	 * 
	 * @param robotsTxtContent
	 * @param userAgent
	 */
	public RobotsPolicy(String robotsTxtContent, String userAgent) {
		String[] rawLines = robotsTxtContent.split("\\R");
		List<String> lines = new ArrayList<>();
		for (String r : rawLines) {
			String c = r.split("#", 2)[0].trim();
			if (!c.isEmpty()) {
				lines.add(c);
			}
		}
		this.rules = new ArrayList<>();
		
		do {
			// We'll try this at most twice, with our actual user-agent and default
			for (int i = 0; i < lines.size(); i++) {
				String s = lines.get(i);
				if (s.startsWith("User-agent:") &&
						s.substring(11).trim().equalsIgnoreCase(userAgent)) {
					// skip over other agents
					int j = i + 1;
					while (j < lines.size() && lines.get(j).startsWith("User-agent:")) {
						j++;
					}
					// read rules
					while (j < lines.size() && !lines.get(j).startsWith("User-agent:")) {
						s = lines.get(j);
						if (s.startsWith("Allow:")) {
							String p = s.substring(6).trim();
							if (p.isEmpty()) {
								this.rules.add(new UrlRule(false));
							} else {
								this.rules.add(new UrlRule(true, p));
							}
						} else if (s.startsWith("Disallow:")) {
							String p = s.substring(9).trim();
							if (p.isEmpty()) {
								this.rules.add(new UrlRule(true));
							} else {
								this.rules.add(new UrlRule(false, p));
							}
						}
						j++;
					}
					break;
				}
			}
			if ("*".equals(userAgent)) {
				break;
			} else {
				userAgent = "*";
			}
		} while (rules.isEmpty());
	}

	public boolean allows(String urlPath) {
		for (UrlRule r : rules) {
			if (r.matches(urlPath)) {
				return r.isAllow();
			}
		}
		return true;
	}

	private static class UrlRule {
		
		/** Constructs a UrlRule that allows or disallows all URL paths. */
		public UrlRule(boolean allow) {
			this.absolute = true;
			this.allow = allow;
		}
		
		public UrlRule(boolean allow, String prefix) {
			this.absolute = false;
			this.allow = allow;
			this.prefix = prefix;
		}
		
		public boolean isAllow() {
			return allow;
		}
		
		public boolean matches(String urlPath) {
			return absolute || urlPath.startsWith(prefix);
		}
		
		private boolean absolute;
		private boolean allow;
		private String prefix;
	}
	
	private List<UrlRule> rules;
}
