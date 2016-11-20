package cosi129.pa3;

import org.apache.mahout.math.Vector;

public class MahoutVector {
	private String classifier;
	private Vector vector;
	
	public MahoutVector(String classifier, Vector vector) {
		this.classifier = classifier;
		this.vector = vector;
	}
	
	public Vector getVector() {
		return vector;
	}
	
	public String getClassifier() {
		return classifier;
	}
}
