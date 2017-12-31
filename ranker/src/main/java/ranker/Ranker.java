package ranker;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public abstract class Ranker {
	
	ArrayList<Integer> features;
	
	/**
	 * Maps the index of an attribute to the classifier it represents.
	 */
	HashMap<Integer,Classifier> classifiersMap;
	ArrayList<Integer> classifierIndices;


	/**
	 * Generates a ranker. 
	 * 
	 * @param data The training data
	 */
	public abstract void buildRanker (Instances data) throws Exception;
	
	public abstract List<Classifier> predictRankingforInstance (Instance instance) throws Exception;
	
	/**
	 * Collects indices of attributes that represent classifiers or meta features.
	 * 
	 * @param data
	 * @throws Exception
	 */
	void getClassifiersAndMetaFeatures(Instances data) throws Exception {
		classifierIndices = new ArrayList<Integer>();
		classifiersMap = new HashMap<Integer,Classifier>();
		features = new ArrayList<Integer>();
		
		// Find the classifiers and meta features
		int labelIndex = 0;
		HashSet<String> portfolio = new HashSet<String>();
		Arrays.asList(Util.portfolio).forEach(classifier->portfolio.add(classifier.getClass().getName()));
		for (int index = 0; index < data.numAttributes(); index++) {
			String attributeName = data.attribute(index).name();
			if (portfolio.contains(attributeName)) {
				classifierIndices.add(index);
				classifiersMap.put(labelIndex, AbstractClassifier.forName(attributeName,null));
				labelIndex++;
			} else {
				features.add(index);
			}
		}
		
		// Check for sensible input
		if (classifierIndices.size() < 2) {
			throw new IllegalArgumentException("Data set given must contain at least two attributes which represent weka classifiers.");
		}
	}
	
	double[] getFeatureValuesForInstance(Instance instance) {
		double[] instanceFeatureValues = new double[features.size()];
		int featureIndex = 0;
		for (int attributeIndex : features) {
			instanceFeatureValues[featureIndex++]=instance.value(attributeIndex);
		}
		return instanceFeatureValues;
	}
}
