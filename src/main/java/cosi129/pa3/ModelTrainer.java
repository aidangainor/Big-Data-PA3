package cosi129.pa3;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.TreeSet;
import java.util.SortedSet;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.classifier.AbstractVectorClassifier;
import org.apache.mahout.classifier.naivebayes.StandardNaiveBayesClassifier;
import org.apache.mahout.classifier.naivebayes.NaiveBayesModel;
import org.apache.mahout.classifier.naivebayes.training.TrainNaiveBayesJob;

public class ModelTrainer {
	private String sequenceFilePath;
	private SortedSet<String> classifications;
	private LemmaVectorizer vectorizer;
	
	/*
	 * Takes in an HDFS path to the directory storing the sequence files and a LemmaVectorizer object
	 */
	public ModelTrainer(String sequenceFilePath, LemmaVectorizer vectorizer) {
		this.sequenceFilePath = sequenceFilePath;
		this.vectorizer = vectorizer;
		this.classifications = new TreeSet<String>();
	}
	
	/*
	 * Takes path string of training data and generates sequence files
	 */
	public void createSequenceFilesFromVectors(String trainingDataPath) 
			throws IOException, URISyntaxException {
		System.out.println("About to write sequence files");
		Configuration conf = new Configuration();
		URI seqFilePath = new URI(this.sequenceFilePath);
		FileSystem fs = FileSystem.get(new Configuration());
		fs.delete(new Path(seqFilePath));
		SequenceFile.Writer writer = SequenceFile.createWriter(fs, conf, new Path(seqFilePath), Text.class, VectorWritable.class);
		String line;
		Path pt = new Path(trainingDataPath);
		BufferedReader fileReader = new BufferedReader(new InputStreamReader(fs.open(pt)));

		while ((line = fileReader.readLine()) != null) {
			MahoutVector vector = vectorizer.vectorizeLemmaLine(line);
			VectorWritable vectorWritable = new VectorWritable();
			vectorWritable.set(vector.getVector());
			String classification = vector.getClassifier();
			// add the classification to our internal set of classifications
			// this is to keep track of *all* professions in a sorted manner for later use
			this.classifications.add(classification);
			writer.append(new Text("/" + classification + "/"), vectorWritable);
		}
		
		fileReader.close();
		writer.close();
	}
	
	/*
	 * Train the model and get it as an AbstractVectorClassifier object
	 */
	public AbstractVectorClassifier getTrainedModel() 
			throws Exception {
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.getLocal(conf);
		
		String outputDirectory = "/aidan/professions/output";
		String tempDirectory = "/aidan/professions/temp";
		
		fs.delete(new Path(outputDirectory), true);
		fs.delete(new Path(tempDirectory), true);
		
		TrainNaiveBayesJob trainNaiveBayes = new TrainNaiveBayesJob();
		trainNaiveBayes.setConf(conf);
		String[] naiveBayesArguments = { "--input", this.sequenceFilePath, "--output", outputDirectory, "--alphaI", "1", "--overwrite", "--tempDir", tempDirectory };
		trainNaiveBayes.run(naiveBayesArguments);
		NaiveBayesModel naiveBayesModel = NaiveBayesModel.materialize(new Path(outputDirectory), conf);

		System.out.println("features: " + naiveBayesModel.numFeatures());
		System.out.println("labels: " + naiveBayesModel.numLabels());
		
	    AbstractVectorClassifier classifier = new StandardNaiveBayesClassifier(naiveBayesModel);
	    return classifier;
	}
	
	/*
	 * Get a sorted array of classifications
	 */
	public String[] getProfessionsList() {
		return classifications.toArray(new String[classifications.size()]);
	}
}