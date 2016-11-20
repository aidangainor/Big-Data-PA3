package cosi129.pa3;

import java.util.ArrayList;

import org.apache.mahout.classifier.AbstractVectorClassifier;
import org.apache.mahout.math.Vector;

public class ModelTester {
	static final String PERSON_LEMMA_INDEX = "";
	static final String SEQUENCE_FILE_PATH = "";
	static final String TRAINING_SET_PATH = "";
	static final String TEST_SET_PATH = "";
	
	public static void testModel(String testSetPath, AbstractVectorClassifier model, LemmaVectorizer lv) 
			throws Exception {
	    ArrayList<MahoutVector> vectors = lv.vectorizeLemmaFile(testSetPath);
	    
	    for (MahoutVector mahoutVector : vectors) {
	    	Vector prediction = model.classifyFull(mahoutVector.getVector());
	    	
	    	// need to figure out how to prediction vector corresponds to classifications
	    	// I'm assuming alphabetical sorting for indexing the vector for each profession
	    }
	}
	
	public static void main(String[] args) throws Exception {
		// Create a lemma vectorizer from the original lemma file
		LemmaVectorizer lv = new LemmaVectorizer(PERSON_LEMMA_INDEX );
		// Set up our model from the full lemma and training data files
		ModelTrainer mt = new ModelTrainer(SEQUENCE_FILE_PATH, lv);
	    mt.initializeTrainingSetVectors(TRAINING_SET_PATH);
	    AbstractVectorClassifier model = mt.getTrainedModel();
	    // Now test it
	    testModel(TEST_SET_PATH, model, lv);
	}
}
