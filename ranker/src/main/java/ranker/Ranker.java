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
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public abstract class Ranker {
	
	ArrayList<Integer> features;
	
	/**
	 * Maps the index of an attribute to the classifier it represents.
	 */
	HashMap<Integer,Classifier> classifiersMap;
	/**
	 * Contains index of classifier attributes in given data set
	 */
	ArrayList<Integer> classifierIndices;


	/**
	 * Generates a ranker. 
	 * 
	 * @param data The training data
	 */
	public abstract void buildRanker (Instances data) throws Exception;
	
	/**
	 * Predicts a ranking of classifiers for the given instance; instance must have the same format as (be compatible with) instances given in buildRanker
	 * 
	 * @param instance
	 * @return
	 * @throws Exception
	 */
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
		HashSet<String> portfolio = new HashSet<String>();
		Arrays.asList(Util.portfolio).forEach(classifier->portfolio.add(classifier.getClass().getName()));
		for (int index = 0; index < data.numAttributes(); index++) {
			String attributeName = data.attribute(index).name();
			if (portfolio.contains(attributeName)) {
				classifierIndices.add(index);
				classifiersMap.put(index, AbstractClassifier.forName(attributeName,null));
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
			double value = instance.value(attributeIndex);
			if (value != Double.NaN) {
				instanceFeatureValues[featureIndex++]= value;				
			} else {
				// TODO remove if worked
				instanceFeatureValues[featureIndex++]= value;
				System.out.println("Found NaN");
			}
			
		}
		return instanceFeatureValues;
	}
}
