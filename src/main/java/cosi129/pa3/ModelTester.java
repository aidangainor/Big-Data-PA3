package cosi129.pa3;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.mahout.classifier.AbstractVectorClassifier;
import org.apache.mahout.math.Vector;

public class ModelTester {
	static String SEQUENCE_FILE_PATH;
	static String TRAINING_SET_PATH;
	static String DICT_PATH;
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
		// Get a mapping from a persons name to their profession from the MapReduce code used earlier
		HashMap<String, ArrayList<String>> personToProfessions = LemmaIndexFormater.getProfessionMapping("professions_test.txt");
		
	    int classifiedCorrect = 0;
	    int classifiedTotal = 0;
	    
		FileSystem fs = FileSystem.get(new Configuration());
		Path pt = new Path(testSetPath);
        FileStatus[] status = fs.listStatus(new Path(testSetPath));
        
        // Get a list of all test data segements (like part-m-0001, part-m-0002, ...)
        for (int fileIndex = 0; fileIndex<status.length; fileIndex++) {
            BufferedReader fileReader = new BufferedReader(new InputStreamReader(fs.open(status[fileIndex].getPath())));
            String line;
            while ((line = fileReader.readLine()) != null) {
    			MahoutVector mahoutVector = lv.vectorizeLemmaLine(line);
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
            fileReader.close();
        }
   
	    double percentCorrect = ((double) classifiedCorrect / classifiedTotal) * 100;
	    System.out.println("\n**********************************");
	    System.out.printf("Percent classified correct = %f\n", percentCorrect);
	    System.out.println("**********************************");
	}
	
	public static void main(String[] args) throws Exception {
		String[] otherArgs = new GenericOptionsParser(args).getRemainingArgs();
		try {
			// This is a path (like dict/part-0000) to a dictionary file of all words
			DICT_PATH = otherArgs[0];
			// Path to training set
			TRAINING_SET_PATH = otherArgs[1];
			// Path to directory which contains test set
			TEST_SET_PATH = otherArgs[2];
			SEQUENCE_FILE_PATH = "/seq_file_container/";
			
			System.out.println("Begginning to run model tester.");
			// Create a lemma vectorizer from the original lemma file
			LemmaVectorizer lv = new LemmaVectorizer(DICT_PATH);
			// Set up our model from the full lemma and training data files
			ModelTrainer mt = new ModelTrainer(SEQUENCE_FILE_PATH, lv);
		    mt.createSequenceFilesFromVectors(TRAINING_SET_PATH);
		    // Now test it
		    testModel(TEST_SET_PATH, mt, lv);
		} catch (ArrayIndexOutOfBoundsException e) {
			System.out.println("Usage: ModelTester dict_path train_set_path test_set_dir");
		}
	}
}

