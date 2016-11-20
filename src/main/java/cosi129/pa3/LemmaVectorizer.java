package cosi129.pa3;

import java.io.IOException;
import java.util.ArrayList;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.HashMap;

import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

import cosi129.pa3.StringIntegerList.StringInteger;

public class LemmaVectorizer {
	// We must keep track of unique words we have found and assign them an incrementing index to keep uniform vectors
	// Also, keep track of the count of unique words (which will be our vector size)
	private HashMap<String, Integer> wordsAndIndexMap;;
	private int vectorSize;
	

	public LemmaVectorizer(String originalLemmaPath) throws IOException {
		this.wordsAndIndexMap = createVectorMappingForAllLemmas(originalLemmaPath);
	}
	
	/*
	 * Returns a mapping for all lemmas found in an an index
	 * This should be run on a universal lemma index, i.e. the combination of both the train and test sets
	 * The reasoning behind this is we must have 100% consistency of our lemma indexing scheme
	 * originalLemmaPath: a string path to the original lemma index of name -> list of lemmas
	 */
	public HashMap<String, Integer> createVectorMappingForAllLemmas(String originalLemmaPath) throws IOException {
		int uniqueWordCount = 0;
		HashMap<String, Integer> wordsAndIndexMap = new HashMap<String, Integer>();
		BufferedReader originalLemmaReader = new BufferedReader(new FileReader(originalLemmaPath));
		String line;
		while ((line = originalLemmaReader.readLine()) != null) {
			StringIntegerList sil = new StringIntegerList();
			sil.readFromString(line);
			for (StringInteger si : sil.getIndices()) {
				String lemma = si.getString();
				if (!wordsAndIndexMap.containsKey(lemma)) {
					uniqueWordCount++;
					wordsAndIndexMap.put(lemma, uniqueWordCount);
				}
			}
		}
		originalLemmaReader.close();
		vectorSize = uniqueWordCount;
		return wordsAndIndexMap;
	}
	
	/*
	 * Take in a path to either training or test set and make a vector of every line in the file
	 */
	public ArrayList<MahoutVector> vectorizeLemmaFile(String pathString) throws IOException {
		ArrayList<MahoutVector> vectors = new ArrayList<MahoutVector>();
		String line;
		BufferedReader fileReader = new BufferedReader(new FileReader(pathString));

		while ((line = fileReader.readLine()) != null) {
			String[] splits = line.split("\t");
			String classifier = splits[0];
			String lemmaIndices = splits[1];
			Vector vector = new RandomAccessSparseVector(vectorSize, vectorSize);
			StringIntegerList sil = new StringIntegerList();
			sil.readFromString(lemmaIndices);
			
			for (StringInteger si : sil.getIndices()) {
				String lemma = si.getString();
				Integer value = si.getValue();
				vector.set(this.wordsAndIndexMap.get(lemma), value);
			}
			MahoutVector mahoutVector = new MahoutVector(classifier, vector);
			vectors.add(mahoutVector);
			
		}
		fileReader.close();
		return vectors;
	}
}