package cosi129.pa3;

import java.util.ArrayList;
import java.util.HashMap;

import org.apache.mahout.classifier.AbstractVectorClassifier;
import org.apache.mahout.math.Vector;

public class ModelTester {
	static final String PERSON_LEMMA_INDEX = "";
	static final String SEQUENCE_FILE_PATH = "";
	static final String TRAINING_SET_PATH = "";
	static final String TEST_SET_PATH = "";
	
	private static class Prediction {
		public double probability;
		public String profession;
		
		public Prediction(String profession, double probability) {
			this.profession = profession;
			this.probability = probability;
		}
	}
	
	public static void testModel(String testSetPath, ModelTrainer mt, LemmaVectorizer lv) 
			throws Exception {
		AbstractVectorClassifier model = mt.getTrainedModel();
		String[] professions = mt.getProfessionsList();
	    ArrayList<MahoutVector> vectors = lv.vectorizeLemmaFile(testSetPath);
		HashMap<String, ArrayList<String>> personToProfessions = LemmaIndexFormater.getProfessionMapping();
		
	    int classifiedCorrect = 0;
	    int classifiedTotal = 0;
	    
	    // Guess every person's profession from the test set
	    for (MahoutVector mahoutVector : vectors) {
	    	// We consider our prediction correct if one of top 3 classifications is correct
	    	Prediction[] top3Predictions = {null, null, null};
	    	Vector predictionVector = model.classifyFull(mahoutVector.getVector());
	    	for (int i = 0; i < predictionVector.size(); i++) {
	    		double probability = predictionVector.get(i);
	    		// Update our current top 3 predictions
	    		for (int j = 0; j < 3; j++) {
	    			Prediction prediction = top3Predictions[j];
	    			if (prediction != null) {
	    				if (probability > prediction.probability) {
	    					String profession = professions[i];
	    					top3Predictions[j] = new Prediction(profession, probability);
	    				}
	    			} else {
	    				String profession = professions[i];
    					top3Predictions[j] = new Prediction(profession, probability);
	    			}
	    		}
	    	}
	    	// Check if we got it right
	    	for (Prediction prediction : top3Predictions) {
	    		ArrayList<String> correctProfessions = personToProfessions.get(prediction.profession);
	    		if (correctProfessions.contains(prediction.profession)) {
	    			classifiedCorrect++;
	    			break;
	    		}
	    	}
	    	classifiedTotal++;
	    }
	    double percentCorrect = ((double) classifiedCorrect / classifiedTotal) * 100;
	    System.out.printf("Percent classified correct = %f\n", percentCorrect);
	}
	
	public static void main(String[] args) throws Exception {
		// Create a lemma vectorizer from the original lemma file
		LemmaVectorizer lv = new LemmaVectorizer(PERSON_LEMMA_INDEX );
		// Set up our model from the full lemma and training data files
		ModelTrainer mt = new ModelTrainer(SEQUENCE_FILE_PATH, lv);
	    mt.createSequenceFilesFromVectors(TRAINING_SET_PATH);
	    // Now test it
	    testModel(TEST_SET_PATH, mt, lv);
	}
}
