package ranker.core.algorithms;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import weka.core.Instance;
import weka.core.Instances;

public abstract class Ranker {

	/**
	 * Data set the ranker is built with.
	 */
	protected Instances data;

	/**
	 * Indices of target attributes in {@link #data}.
	 */
	protected List<Integer> targetAttributes;

	/**
	 * Indices of meta feature attributes in {@link #data}.
	 */
	protected List<Integer> features;

	/**
	 * Maps the index of an attribute to the classifier it represents.
	 */
	protected Map<Integer, String> classifiersMap;

	/**
	 * Generates a ranker.
	 * 
	 * @param data
	 *            The training data
	 * @param targetAttributes
	 *            The target attributes
	 * @throws Exception if the ranker encounters a problem being built
	 */
	public void buildRanker(Instances data, List<Integer> targetAttributes) throws Exception {
		// Check for sensible input
		if (targetAttributes.size() < 2) {
			throw new IllegalArgumentException(
					"Data set given must contain at least two attributes which represent weka classifiers.");
		}

		// Initialize variables
		this.data = data;
		this.targetAttributes = targetAttributes;
		classifiersMap = new HashMap<Integer, String>();
		features = new ArrayList<Integer>();

		for (int i = 0; i < data.numAttributes(); i++) {
			String attributeName = data.attribute(i).name();
			if (targetAttributes.contains(i)) {
				classifiersMap.put(i, attributeName);
			} else {
				features.add(i);
			}
		}

		initialize();
	}

	/**
	 * Predicts a ranking of classifiers for the given instance. Instance must have
	 * the same format as (be compatible with) instances given in buildRanker, i.e.
	 * contain the same attributes in the same order. Target attributes need not
	 * have values but must exist.
	 * 
	 * @param instance
	 *            The instance for which to predict a ranking
	 * @return A ranking of learning algorithms
	 * @throws Exception
	 *             If a prediction cannot be made
	 */
	public abstract List<String> predictRankingforInstance(Instance instance) throws Exception;

	/**
	 * Returns a list with the exact performance values predicted for each
	 * algorithm. The value at position i in the list is the predicted value for the
	 * model at position i in the returned ranking. May cause an
	 * {@link java.lang.UnsupportedOperationException#UnsupportedOperationException}
	 * if the implementation of the Ranker does not internally try to predict
	 * performance values for the learning algorithms.
	 * 
	 * @return The estimated values for the models, null if no prediction has been
	 *         made so far.
	 */
	public abstract List<Double> getEstimates();

	/**
	 * Has to initialize the ranker so that it is able to return predictions for a
	 * new instance.
	 * 
	 * @throws Exception
	 *             If building the ranker failed
	 */
	protected abstract void initialize() throws Exception;

	/**
	 * Compiles the feature values for that instance to a double array. Assumes that
	 * the instance contains both the meta feature and target attributes still.
	 * 
	 * @param instance
	 *            The instance of which to get the features
	 * @return The feature values of the instance
	 */
	protected double[] getFeatureValuesForInstance(Instance instance) {
		double[] instanceFeatureValues = new double[features.size()];
		int featureIndex = 0;
		for (int attributeIndex : features) {
			double value = instance.value(attributeIndex);
			instanceFeatureValues[featureIndex++] = value;
		}
		return instanceFeatureValues;
	}

	public String getName() {
		return this.getClass().getSimpleName();
	}

	public Object getClassifierString() {
		return this.getClass().getSimpleName();
	}
}
