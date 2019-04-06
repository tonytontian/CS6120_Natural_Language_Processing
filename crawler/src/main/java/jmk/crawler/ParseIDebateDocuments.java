package jmk.crawler;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;
import java.io.Reader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;

import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.process.DocumentPreprocessor;

public class ParseIDebateDocuments {
	
	public static class Debate {
		public Debate(String name, List<Claim> supportingClaims, List<Claim> opposingClaims) {
			this.name = name;
			this.supportingClaims = supportingClaims;
			this.opposingClaims = opposingClaims;
		}
		
		public String name;
		public List<Claim> supportingClaims;
		public List<Claim> opposingClaims;
	}
	
	public static class Claim {
		public Claim(String claim) {
			this.claim = claim;
		}
		
		public String claim;
		public String argument;
		public String counter;
	}

	public static void main(String[] args) {
		List<Debate> debates = parseDebateHtml();
		saveDebatesJson(debates);
	}
	
	public static List<Debate> parseDebateHtml() {
		DocStore docStore = new DocStore(new File(Crawler.DOCSTORE_PATH));
		List<Debate> debates = new ArrayList<>();
		for (String url : docStore.getUrlSet()) {
			Document doc = loadDocumentFromStore(url, docStore);
			if (doc == null) {
				continue;
			}
			Element debateDiv = doc.selectFirst("div.node-debatabase");
			if (debateDiv == null) {
				//System.out.println("Not a debate page: " + url);
				continue;
			}
			String debateName = debateDiv.selectFirst("div.debatabase-title").text();
			Element supportingDiv = debateDiv.selectFirst("div#debatabase-points-1");
			List<Claim> supportingClaims = new ArrayList<>();
			for (Element claimDiv : supportingDiv.select("div.content")) {
				Element div = claimDiv.selectFirst("div.field-name-field-title-point-for .field-item");
				if (div == null) {
					continue;
				}
				Claim claim = new Claim(div.text());
				div = claimDiv.selectFirst("div.field-name-field-point-point-for .field-item");
				if (div != null) {
					claim.argument = div.text();
				}
				div = claimDiv.selectFirst("div.field-name-field-counterpoint-point-for .field-item");
				if (div != null) {
					claim.counter = div.text();
				}
				supportingClaims.add(claim);
			}
			Element opposingDiv = debateDiv.selectFirst("div#debatabase-points-2");
			List<Claim> opposingClaims = new ArrayList<>();
			for (Element claimDiv : opposingDiv.select("div.content")) {
				Element div = claimDiv.selectFirst("div.field-name-field-title .field-item");
				if (div == null) {
					continue;
				}
				Claim claim = new Claim(div.text());
				div = claimDiv.selectFirst("div.field-name-field-point .field-item");
				if (div != null) {
					claim.argument = div.text();
				}
				div = claimDiv.selectFirst("div.field-name-field-counterpoint .field-item");
				if (div != null) {
					claim.counter = div.text();
				}
				opposingClaims.add(claim);
			}
			Debate d = new Debate(debateName, supportingClaims, opposingClaims);
			debates.add(d);
			//printDebate(d);
		}
		return debates;
	}
	
	public static void printDebate(Debate d) {
		System.out.println(d.name);
		for (Claim claim : d.supportingClaims) {
			System.out.println("  for: " + claim.claim);
			if (claim.argument != null) {
				String[] bodyAndRefs = claim.argument.split(" 1: ");
				for (String arg : splitArgument(bodyAndRefs[0])) {
					System.out.println("    argument: " + arg);
				}
			}
			if (claim.counter != null) {
				String[] bodyAndRefs = claim.counter.split(" 1: ");
				for (String arg : splitArgument(bodyAndRefs[0])) {
					System.out.println("    counter: " + arg);
				}
			}
		}
		for (Claim claim : d.opposingClaims) {
			System.out.println("  against: " + claim.claim);
			if (claim.argument != null) {
				String[] bodyAndRefs = claim.argument.split(" 1: ");
				for (String arg : splitArgument(bodyAndRefs[0])) {
					System.out.println("    argument: " + arg);
				}
			}
			if (claim.counter != null) {
				String[] bodyAndRefs = claim.counter.split(" 1: ");
				for (String arg : splitArgument(bodyAndRefs[0])) {
					System.out.println("    counter: " + arg);
				}
			}
		}
	}
	
	public static void saveDebatesJson(List<Debate> debates) {
		int debateId = 1000000;
		boolean first = true;
		try (PrintStream out = new PrintStream("debates.json")) {
			out.print("[");
			for (Debate d : debates) {
				int claimId = 0;
				for (Claim c : d.supportingClaims) {
					if (first) {
						first = false;
					} else {
						out.print(',');
					}
					writeClaim(c, claimId, false, debateId, d.name, out);
					claimId++;
				}
				for (Claim c : d.opposingClaims) {
					if (first) {
						first = false;
					} else {
						out.print(',');
					}
					writeClaim(c, claimId, true, debateId, d.name, out);
					claimId++;
				}
				debateId++;
			}
			out.print("]");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	
	private static void writeClaim(Claim c, int claimId, boolean opposing,
			int debateId, String debateName, PrintStream out) {
		out.println("{");
		writeField(out, "_debate_id", String.valueOf(debateId));
		writeField(out, "_claim", c.claim.replaceAll("\"", ""));
		writeField(out, "_is_opposing", String.valueOf(opposing));
		out.println("\"_argument_sentences\": {");
		int sId = 0;
		List<String> ss = new ArrayList<>();
		if (c.argument != null) {
			for (String s : splitArgument(c.argument)) {
				ss.add(field(String.format("%d_%d_%d", debateId, claimId, sId), s));
				sId++;
			}
		}
		out.println(String.join(",\n", ss));
		out.println("},");
		out.println("\"_counter_sentences\": {");
		ss = new ArrayList<>();
		if (c.counter != null) {
			for (String s : splitArgument(c.counter)) {
				ss.add(field(String.format("%d_%d_%d", debateId, claimId, sId), s));
				sId++;
			}
		}
		out.println(String.join(",\n", ss));
		out.println("},");
		writeField(out, "_debate_name", debateName.replaceAll("\"", ""));
		writeFieldLast(out, "_claim_id", String.format("%d_%d", debateId, claimId));
		out.println("}");
	}
	
	private static String field(String key, String value) {
		return String.format("  \"%s\": \"%s\"", key, value);
	}
	
	private static void writeField(PrintStream out, String key, String value) {
		out.printf("  \"%s\": \"%s\",\n", key, value);
	}
	
	private static void writeFieldLast(PrintStream out, String key, String value) {
		out.printf("  \"%s\": \"%s\"\n", key, value);
	}
	
	private static List<String> splitArgument(String argument) {
		List<String> ss = new ArrayList<>();
		Reader r = new StringReader(argument);
		DocumentPreprocessor pre = new DocumentPreprocessor(r);
		for (List<HasWord> sent : pre) {
			if (sent.size() <= 1) {
				continue;
			}
			ss.add(sent.stream().map(HasWord::word)
					.filter(w -> !w.startsWith("http:"))
					.map(w -> w.replace("\"", ""))
					.collect(Collectors.joining(" ")));
		}
		return ss;
	}
	
	/** Returns null on failure */
	private static Document loadDocumentFromStore(String url, DocStore docStore) {
		String charset = "UTF-8";
		byte[] docBytes = docStore.getDocumentContent(url);
		InputStream in = new ByteArrayInputStream(docBytes);
		try {
			return Jsoup.parse(in, charset, url);
		} catch (IOException e) {
			System.err.println("Could not parse stored document: " + url);
			return null;
		}
	}

}
