package ranker.core.algorithms.decomposition;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import ranker.core.algorithms.Ranker;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Ranks algorithm by decomposing the problem into predicting a value for every
 * single item to be ranked and then ranking by the predicted values.
 * 
 * @author helegraf
 *
 */
public abstract class DecompositionRanker extends Ranker {

	/**
	 * Maps learning algorithms to their models.
	 */
	protected Map<String, Classifier> models;

	/**
	 * Maps learning algorithms to their training data.
	 */
	protected HashMap<String, Instances> trainingData;

	/**
	 * Contains estimates of learning algorithm performances for the last predicted
	 * ranking.
	 */
	protected List<Double> estimates;

	@Override
	public List<String> predictRankingforInstance(Instance instance) throws Exception {
		// Initialize data structures to save results
		TreeMap<Double, List<String>> predictions = new TreeMap<>();
		ArrayList<String> results = new ArrayList<>();

		// Construct querying instance
		double[] newFeatures = new double[features.size() + 1];
		for (int i : features) {
			newFeatures[i] = instance.value(i);
		}
		newFeatures[newFeatures.length - 1] = Double.NaN;

		// Calculate results
		for (String item : models.keySet()) {
			Instance newInstance = new DenseInstance(newFeatures.length, newFeatures);
			newInstance.setDataset(trainingData.get(item));
			double result = models.get(item).classifyInstance(newInstance);
			if (predictions.containsKey(result)) {
				predictions.get(result).add(item);
			} else {
				ArrayList<String> items = new ArrayList<>();
				items.add(item);
				predictions.put(result, items);
			}
		}

		// Build list of results
		estimates = new ArrayList<>();
		predictions.descendingMap().forEach((value, classifierList) -> {
			results.addAll(classifierList);
			classifierList.forEach(classifier -> estimates.add(value));
		});
		return results;
	}

	/**
	 * Builds a model for each classifier. Has to initialize the list of models
	 * 
	 * @param train the training data for each item to be ranked
	 * @throws Exception if something goes wrong while building the models
	 */
	protected abstract void buildModels(Map<String, Instances> train) throws Exception;

	@Override
	protected void initialize() throws Exception {
		trainingData = new HashMap<>();

		// Get all attributes
		ArrayList<Attribute> attributes = new ArrayList<>();
		for (int i : features) {
			attributes.add(new Attribute(data.attribute(i).name()));
		}
		attributes.add(new Attribute("Performance"));

		// Create new Instances for each target attribute
		for (int i : targetAttributes) {
			trainingData.put(classifiersMap.get(i), new Instances(classifiersMap.get(i), attributes, 0));
			trainingData.get(classifiersMap.get(i)).setClassIndex(features.size());
		}

		for (int i = 0; i < data.numInstances(); i++) {
			modifyInstance(data.get(i), targetAttributes);
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
				trainingData.get(classifiersMap.get(i))
						.add(new DenseInstance(newFeatureValues.length, newFeatureValues));
			}
		}

		buildModels(trainingData);
	}

	/**
	 * Adds the possibility for instance modification before learning, e.g.
	 * replacing regression values by ranks.
	 * 
	 * @param instance         the instance to modify
	 * @param targetAttributes the target attributes of the instance
	 */
	protected void modifyInstance(Instance instance, List<Integer> targetAttributes) {
		// by default, no modification is done
	}

	@Override
	public List<Double> getEstimates() {
		return estimates;
	}
}
