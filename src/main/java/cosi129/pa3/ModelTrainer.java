package cosi129.pa3;

import java.io.IOException;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.classifier.AbstractVectorClassifier;
import org.apache.mahout.classifier.naivebayes.ComplementaryNaiveBayesClassifier;
import org.apache.mahout.classifier.naivebayes.NaiveBayesModel;
import org.apache.mahout.classifier.naivebayes.training.TrainNaiveBayesJob;

public class ModelTrainer {
	String sequenceFilePath;
	LemmaVectorizer vectorizer;
	/*
	 * Takes in an HDFS path to the directory storing the sequence files and a LemmaVectorizer object
	 */
	public ModelTrainer(String sequenceFilePath, LemmaVectorizer vectorizer) {
		this.sequenceFilePath = sequenceFilePath;
		this.vectorizer = vectorizer;
	}
	
	/*
	 * Takes in two path strings and creates vectors for the training data
	 * fullLemmaIndexPath: The path for both the training data and 
	 */
	public void initializeTrainingSetVectors(String trainingDataPath) throws IOException {
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.getLocal(conf);
		Path seqFilePath = new Path(this.sequenceFilePath);
		fs.delete(seqFilePath, false);
		SequenceFile.Writer writer = SequenceFile.createWriter(fs, conf, seqFilePath, Text.class, VectorWritable.class);
		List<MahoutVector> vectors = vectorizer.vectorizeLemmaFile(trainingDataPath);
		
		for (MahoutVector vector : vectors) {
			VectorWritable vectorWritable = new VectorWritable();
			vectorWritable.set(vector.getVector());
			writer.append(new Text("/" + vector.getClassifier() + "/"), vectorWritable);
		}
		writer.close();
	}
	
	/*
	 * Train the model and get it as an AbstractVectorClassifier object
	 */
	public AbstractVectorClassifier getTrainedModel() throws Exception {
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.getLocal(conf);
		
		String outputDirectory = "/aidan/professions/output";
		String tempDirectory = "/aidan/professions/temp";
		
		fs.delete(new Path(outputDirectory), true);
		fs.delete(new Path(tempDirectory), true);
		
		TrainNaiveBayesJob trainNaiveBayes = new TrainNaiveBayesJob();
		trainNaiveBayes.setConf(conf);
		String[] naiveBayesArguments = { "--input", this.sequenceFilePath, "--output", outputDirectory, "-el", "--overwrite", "--tempDir", tempDirectory };
		trainNaiveBayes.run(naiveBayesArguments);
		NaiveBayesModel naiveBayesModel = NaiveBayesModel.materialize(new Path(outputDirectory), conf);

		System.out.println("features: " + naiveBayesModel.numFeatures());
		System.out.println("labels: " + naiveBayesModel.numLabels());
		
	    AbstractVectorClassifier classifier = new ComplementaryNaiveBayesClassifier(naiveBayesModel);
	    return classifier;
	}
}