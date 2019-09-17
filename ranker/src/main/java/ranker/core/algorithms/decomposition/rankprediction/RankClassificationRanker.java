package ranker.core.algorithms.decomposition.rankprediction;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import ranker.core.algorithms.decomposition.conflictresolution.ConflictResolutionStrategy;
import ranker.core.algorithms.decomposition.conflictresolution.ExpectedRankStrategy;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

/**
 * Predicts a ranking of algorithms by predicting the rank of each algorithm
 * separately, posed as a classification problem. A conflict resolutions
 * strategy is used for algorithms for which the same rank has been predicted.
 *
 * @author helegraf
 *
 */
public class RankClassificationRanker extends RankRegressionRanker {

	private ConflictResolutionStrategy conflictResolutionStrategy = new ExpectedRankStrategy();

	/**
	 * Construct a RankClassificationRanker with the given name (fully qualified name of
	 * a weka classifier which will be used for predictions).
	 * 
	 * @param name fully qualified name of a weka classifier
	 */
	public RankClassificationRanker(String name) {
		super(name);
	}
	
	public RankClassificationRanker(String name, String[] hyperparameters) {
		super(name, hyperparameters);
	}

	@Override
	protected void buildModels(Map<String, Instances> train) throws Exception {
		for (Map.Entry<String, Instances> entry : train.entrySet()) {
			NumericToNominal filter = new NumericToNominal();
			filter.setOptions(new String[] { "-R", "last" });
			filter.setInputFormat(entry.getValue());
			entry.setValue(Filter.useFilter(entry.getValue(), filter));
		}

		super.buildModels(train);
	}

	@Override
	public List<String> predictRankingforInstance(Instance instance) throws Exception {
		// Initialize data structures to save results
		TreeMap<Double, List<String>> predictions = new TreeMap<>();

		// Construct querying instance
		double[] newFeatures = new double[features.size() + 1];
		for (int i : features) {
			newFeatures[i] = instance.value(i);
		}
		newFeatures[newFeatures.length - 1] = Double.NaN;

		HashMap<String, double[][]> distributions = new HashMap<>();

		// Calculate results
		for (String item : models.keySet()) {
			Instance newInstance = new DenseInstance(newFeatures.length, newFeatures);
			newInstance.setDataset(trainingData.get(item));
			int resultindex = (int) models.get(item).classifyInstance(newInstance);
			double[] distribution = models.get(item).distributionForInstance(newInstance);

			Enumeration<Object> attVals = trainingData.get(item).attribute(trainingData.get(item).classIndex())
					.enumerateValues();

			double[] values = Collections.list(attVals).stream().mapToDouble(o -> {
				return Double.parseDouble((String) o);
			}).toArray();

			double result = values[resultindex];
			distributions.put(item, new double[][] { distribution, values });

			if (predictions.containsKey(result)) {
				predictions.get(result).add(item);
			} else {
				ArrayList<String> classifiers = new ArrayList<>();
				classifiers.add(item);
				predictions.put(result, classifiers);
			}
		}

		conflictResolutionStrategy.resolveConflictsAmongPredictions(predictions, distributions);

		// Build list of results
		ArrayList<String> results = new ArrayList<>();
		estimates = new ArrayList<>();
		predictions.descendingMap().forEach((value, classifierList) -> {
			results.addAll(classifierList);
			classifierList.forEach(classifier -> estimates.add(value));
		});
		return results;
	}

	public ConflictResolutionStrategy getConflictResolutionStrategy() {
		return conflictResolutionStrategy;
	}

	public void setConflictResolutionStrategy(ConflictResolutionStrategy conflictResolutionStrategy) {
		this.conflictResolutionStrategy = conflictResolutionStrategy;
	}

}
