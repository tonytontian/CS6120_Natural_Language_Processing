package jmk.crawler;


import java.io.BufferedReader;
import java.io.Closeable;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.io.RandomAccessFile;
import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * A DocStore is a collection of crawled documents in raw HTML form.
 * It is represented on disk by a directory containing:
 *
 *   a) a contents file, in which each line includes, as tab-separated text:
 *       1) the original URL of a document
 *       2) the time the document was retrieved, in ISO 8601 format 
 *       3) the length of the retrieved content
 *       4) the name of the file in which it is stored locally, relative
 *          to the document store directory
 *       5) the offset within that file
 *       6) the content type of the document, possibly including a charset
 *   b) a collection of data files, each storing a number of downloaded
 *      documents.
 *
 * Note that this representation does not specify a policy for assigning
 * new documents to data files.  This is intentional, to give flexibility
 * in changing the policy as the document store grows.
 * 
 * At the present time, there is no mechanism for garbage collection.  If
 * the content for a document is updated, the space used for the old version
 * becomes inaccessible.
 * 
 * This class is not thread-safe.  It does make reasonable efforts to
 * maintain data consistency after a crash.
 *
 * @author Justin
 *
 */
public class DocStore implements Closeable {
	
	public DocStore(File directory) {
		if (!directory.isDirectory() || !directory.canWrite()) {
			throw new IllegalArgumentException("not a writeable directory");
		}
		this.directory = directory;
		this.docs = new HashMap<>();
		this.dataFiles = new HashMap<>();
		File contents = new File(directory.toString() + File.separator + "contents.txt");
		if (contents.exists()) {
			readContents(contents);
		}
		try {
			contentsPrintStream =
					new PrintStream(new FileOutputStream(contents, true),
							true);
		} catch (FileNotFoundException e) {
			throw new RuntimeException(e);
		}
	}
	
	public Set<String> getUrlSet() {
		return Collections.unmodifiableSet(new HashSet<>(docs.keySet()));
	}
	
	public boolean hasDocument(String url) {
		return docs.containsKey(url);
	}
	
	public String getDocumentContentType(String url) {
		return docs.get(url).contentType;
	}
	
	public Date getDocumentRetrievalDate(String url) {
		return docs.get(url).retrieved;
	}
	
	public byte[] getDocumentContent(String url) {
		try {
			DocDescriptor doc = docs.get(url);
			RandomAccessFile f = getOpenDataFile(doc.file);
			byte[] content = new byte[(int) doc.length];
			f.seek(doc.offset);
			f.readFully(content);
			return content;
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	public void saveDocument(String url, byte[] content, Date retrieved, String contentType) {
		try {
			String dataFileName = chooseDataFileForNewDoc(url);
			RandomAccessFile f = getOpenDataFile(dataFileName);
			long offset = f.length();
			f.seek(offset);
			f.write(content);
			DocDescriptor doc = new DocDescriptor();
			doc.url = url;
			doc.retrieved = retrieved;
			doc.length = content.length;
			doc.file = dataFileName;
			doc.offset = offset;
			doc.contentType = contentType;
			contentsPrintStream.println(doc.toString());
			docs.put(url, doc);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}
	
	public void close() {
		dataFiles.values().stream().forEach(f -> {
			try {
				f.close();
			} catch (IOException e) {
				// Meh
			}
		});
		contentsPrintStream.close();
	}
	
	/**
	 * The fields in this class correspond to the values that make up
	 * a line of the contents file, as described above.
	 */
	private static class DocDescriptor {
		String url;
		Date retrieved;
		long length;
		String file;
		long offset;
		String contentType;
		
		public String toString() {
			return url + "\t" + isoDateFormat.format(retrieved) + "\t"
					+ length + "\t" + file + "\t" + offset + "\t" + contentType;
		}
	}
	
	private static final DateFormat isoDateFormat = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSZ");

	private void readContents(File contents) {
		try (BufferedReader br = new BufferedReader(new FileReader(contents))) {
			br.lines().forEach(line -> {
				String[] parts = line.split("\t");
				DocDescriptor doc = new DocDescriptor();
				doc.url = parts[0];
				try {
					doc.retrieved = isoDateFormat.parse(parts[1]);
				} catch (ParseException e) {
					// This should never happen, because we use this format to output
					throw new RuntimeException(e);
				}
				doc.length = Long.parseLong(parts[2]);
				doc.file = parts[3];
				doc.offset = Long.parseLong(parts[4]);
				doc.contentType = parts[5];
				docs.put(doc.url, doc);
			});
		} catch (IOException e) {
			// This should never happen, because we know it exists
			throw new RuntimeException(e);
		}
	}
	
	private String chooseDataFileForNewDoc(String url) {
		// Use few enough data files that we can keep them all open at once
		String suffix = Integer.toHexString((url.hashCode() >>> 1) % 16);
		return "data_" + suffix + ".txt";
	}
	
	private RandomAccessFile getOpenDataFile(String name) {
		if (!dataFiles.containsKey(name)) {
			String path = directory.toString() + File.separator + name;
			try {
				RandomAccessFile f = new RandomAccessFile(path, "rwd");
				dataFiles.put(name, f);
			} catch (FileNotFoundException e) {
				throw new RuntimeException(e);
			}
		}
		return dataFiles.get(name);
	}
	
	/** Location of the store on disk */
	private File directory;
	
	/** Mapping from URL to metadata */
	private Map<String, DocDescriptor> docs;
	
	/** Mapping from data file relative pathname to open file. */
	private Map<String, RandomAccessFile> dataFiles;
	
	/** Contents file. We keep this open to accommodate additional
	 *  documents. */
	private PrintStream contentsPrintStream;
}
