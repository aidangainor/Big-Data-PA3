package cosi129.pa3;

import java.io.IOException;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.HashMap;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

import cosi129.pa3.StringIntegerList.StringInteger;

public class LemmaVectorizer {
	// We must keep track of unique words we have found and assign them an incrementing index to keep uniform vectors
	// Also, keep track of the count of unique words (which will be our vector size)
	private HashMap<String, Integer> wordsAndIndexMap;;
	private int vectorSize;
	

	public LemmaVectorizer(String originalLemmaPath) 
			throws IOException {
		this.wordsAndIndexMap = createVectorMappingForAllLemmas(originalLemmaPath);
	}
	
	/*
	 * Returns a mapping for all lemmas found in an an index
	 * This should be run on a universal lemma index, i.e. the combination of both the train and test sets
	 * The reasoning behind this is we must have 100% consistency of our lemma indexing scheme
	 * originalLemmaPath: a string path to the original lemma index of name -> list of lemmas
	 */
	public HashMap<String, Integer> createVectorMappingForAllLemmas(String originalLemmaPath) 
			throws IOException {
		int uniqueWordCount = 0;
		FileSystem fs = FileSystem.get(new Configuration());
		HashMap<String, Integer> wordsAndIndexMap = new HashMap<String, Integer>();
		System.out.println("lemma path = " + originalLemmaPath);
		Path pt = new Path(originalLemmaPath);
		BufferedReader originalLemmaReader = new BufferedReader(new InputStreamReader(fs.open(pt)));
		String line;
		while ((line = originalLemmaReader.readLine()) != null) {
			StringIntegerList sil = new StringIntegerList();
			sil.readFromString(line);
			for (StringInteger si : sil.getIndices()) {
				String lemma = si.getString();
				if (!wordsAndIndexMap.containsKey(lemma)) {
					wordsAndIndexMap.put(lemma, uniqueWordCount);
					uniqueWordCount++;
				}
			}
		}
		originalLemmaReader.close();
		vectorSize = uniqueWordCount;
		return wordsAndIndexMap;
	}
	
	/*
	 * Takes in a line of a file and turns it into a MahoutVector
	 */
	public MahoutVector vectorizeLemmaLine(String fileLine) 
			throws IOException {

		String[] splits = fileLine.split("\t");
		String classifier = splits[0];
		String lemmaIndices = splits[1];
		Vector vector = new RandomAccessSparseVector(vectorSize, vectorSize);
		
		StringIntegerList sil = new StringIntegerList();
		sil.readFromString(lemmaIndices);
		
		for (StringInteger si : sil.getIndices()) {
			String lemma = si.getString();
			Integer value = si.getValue();
			Integer vectorIndex = this.wordsAndIndexMap.get(lemma);
			// If vector index is null, this means for some reason a lemma appeared in training data but not full index ?
			if (vectorIndex != null) {
				vector.set(vectorIndex, value);
			}
		}
		return new MahoutVector(classifier, vector);
	}
}