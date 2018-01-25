package ranker.algorithms;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public abstract class RegressionRanker extends Ranker {

	/**
	 * Maps that contains a regression model for each classifier
	 * (classifier -> regression model)
	 */
	protected Map<Classifier, Classifier> regressionModels;

	/**
	 * Map that contains a training data set for each regression model.
	 */
	protected HashMap<Classifier, Instances> map;
	
	/**
	 * List with estimates of learning algorithm performances for last predicted
	 * ranking.
	 */
	protected List<Double> estimates;

	@Override
	public List<Classifier> predictRankingforInstance(Instance instance) throws Exception {
		// Initialize data structures to save results
		TreeMap<Double, List<Classifier>> predictions = new TreeMap<Double, List<Classifier>>();
		ArrayList<Classifier> results = new ArrayList<Classifier>();

		// Construct querying instance
		double[] newFeatures = new double[features.size() + 1];
		for (int i : features) {
			newFeatures[i] = instance.value(i);
		}
		newFeatures[newFeatures.length - 1] = Double.NaN;

		// Calculate results
		for (Classifier classifier : regressionModels.keySet()) {
			Instance newInstance = new DenseInstance(newFeatures.length, newFeatures);
			newInstance.setDataset(map.get(classifier));
			double result = regressionModels.get(classifier).classifyInstance(newInstance);
			if (predictions.containsKey(result)) {
				predictions.get(result).add(classifier);
			} else {
				ArrayList<Classifier> classifiers = new ArrayList<Classifier>();
				classifiers.add(classifier);
				predictions.put(result, classifiers);
			}
		}

		// Build list of results
		estimates = new ArrayList<Double>();
		predictions.descendingMap().forEach((value, classifierList) -> {
			results.addAll(classifierList);
			classifierList.forEach(classifier -> estimates.add(value));
		});
		return results;
	}

	@Override
	public List<Double> getEstimates() {
		return estimates;
	}

	/**
	 * Builds a regression model for each classifier. Has to initialize the list of
	 * regressionModels
	 * 
	 * @param train
	 * @throws Exception
	 */
	protected abstract void buildRegressionModels(Map<Classifier, Instances> train) throws Exception;

	@Override
	protected void initialize() throws Exception {
		// Initialize variable
		map = new HashMap<Classifier, Instances>();

		// Get all attributes
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		for (int i : features) {
			attributes.add(new Attribute(data.attribute(i).name()));
		}
		attributes.add(new Attribute("Performance"));

		// Create new Instances for each target attribute
		for (int i : targetAttributes) {
			map.put(classifiersMap.get(i), new Instances(classifiersMap.get(i).getClass().getName(), attributes, 0));
			map.get(classifiersMap.get(i)).setClassIndex(features.size());
		}

		// Add each Instance of given train set to each of the individual train set with
		// corresponding learning algorithm predictions
		for (Instance instance : data) {
			double[] featureValues = getFeatureValuesForInstance(instance);
			for (int i : targetAttributes) {
				double[] newFeatureValues = new double[featureValues.length + 1];
				for (int j = 0; j < featureValues.length; j++) {
					newFeatureValues[j] = featureValues[j];
				}
				newFeatureValues[featureValues.length] = instance.value(i);
				map.get(classifiersMap.get(i)).add(new DenseInstance(newFeatureValues.length, newFeatureValues));
			}
		}

		buildRegressionModels(map);
	}
}
