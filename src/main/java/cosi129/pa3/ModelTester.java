package cosi129.pa3;

import java.util.ArrayList;
import java.util.HashMap;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.mahout.classifier.AbstractVectorClassifier;
import org.apache.mahout.math.Vector;

public class ModelTester {
	static String PERSON_LEMMA_INDEX;
	static String SEQUENCE_FILE_PATH;
	static String TRAINING_SET_PATH;
	static String TEST_SET_PATH;
	
	private static class Prediction {
		public double probability;
		public String profession;
		public String name;
		
		public Prediction(String profession, String name, double probability) {
			this.profession = profession;
			this.name = name;
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
	    	String name = mahoutVector.getClassifier();
	    	for (int i = 0; i < predictionVector.size(); i++) {
	    		double probability = predictionVector.get(i);
	    		// Update our current top 3 predictions
	    		for (int j = 0; j < 3; j++) {
	    			Prediction prediction = top3Predictions[j];
	    			if (prediction != null) {
	    				if (probability > prediction.probability) {
	    					String profession = professions[i];
	    					top3Predictions[j] = new Prediction(profession, name, probability);
	    					break;
	    				}
	    			} else {
	    				String profession = professions[i];
    					top3Predictions[j] = new Prediction(profession, name, probability);
    					break;
	    			}
	    		}
	    	}
	    	System.out.println("Name = " + name);
	    	
	    	for (Prediction prediction : top3Predictions) {
	    		System.out.println("Prediction prof : " + prediction.profession);
	    	}
	    	// Check if we got it right
	    	for (Prediction prediction : top3Predictions) {
	    		ArrayList<String> correctProfessions = personToProfessions.get(prediction.name);
	    		System.out.println("Correct prof = " + correctProfessions);
	    		// If the list of correct professions is null, this mean test data contains people without labeled profession
	    		if (correctProfessions != null) {
		    		if (correctProfessions.contains(prediction.profession)) {
		    			classifiedCorrect++;
		    			break;
		    		}
	    		} else {
	    			// This means not labeled profession, so we ignore this from classification result
	    			classifiedTotal--;
	    			break;
	    		}
	    	}
	    	System.out.println("");
	    	classifiedTotal++;
	    }
	    System.out.println("AM.HOLMES CHECK " + personToProfessions.get("A. M. Homes"));
	    double percentCorrect = ((double) classifiedCorrect / classifiedTotal) * 100;
	    System.out.printf("Percent classified correct = %f\n", percentCorrect);
	}
	
	public static void main(String[] args) throws Exception {
		String[] otherArgs = new GenericOptionsParser(args).getRemainingArgs();
		try {
			PERSON_LEMMA_INDEX = otherArgs[0];
			TRAINING_SET_PATH = otherArgs[1];
			TEST_SET_PATH = otherArgs[2];
			SEQUENCE_FILE_PATH = "/seq_file_container/";
			
			System.out.println("Attempting to run test");
			// Create a lemma vectorizer from the original lemma file
			LemmaVectorizer lv = new LemmaVectorizer(PERSON_LEMMA_INDEX );
			// Set up our model from the full lemma and training data files
			ModelTrainer mt = new ModelTrainer(SEQUENCE_FILE_PATH, lv);
		    mt.createSequenceFilesFromVectors(TRAINING_SET_PATH);
		    // Now test it
		    testModel(TEST_SET_PATH, mt, lv);
		} catch (ArrayIndexOutOfBoundsException e) {
			System.out.println("Usage: ModelTester lemma_index train_set test_set output_seq_file");
		}
	}
}

