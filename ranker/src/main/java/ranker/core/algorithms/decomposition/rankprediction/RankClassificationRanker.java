package ranker.core.algorithms.decomposition.rankprediction;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import ranker.core.algorithms.decomposition.conflictresolution.ConflictResolutionStrategy;
import ranker.core.algorithms.decomposition.conflictresolution.MeanRankStrategy;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class RankClassificationRanker extends RankRegressionRanker {
	
	private ConflictResolutionStrategy conflictResolutionStrategy = new MeanRankStrategy();

	@Override
	protected void buildRegressionModels(Map<String, Instances> train) throws Exception {
		for (Map.Entry<String, Instances> entry : train.entrySet()) {
			NumericToNominal filter = new NumericToNominal();
			filter.setOptions(new String[] { "-R","last" });
			filter.setInputFormat(entry.getValue());
			entry.setValue(Filter.useFilter(entry.getValue(), filter));
		}

		super.buildRegressionModels(train);
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
		for (String item : regressionModels.keySet()) {
			Instance newInstance = new DenseInstance(newFeatures.length, newFeatures);
			newInstance.setDataset(trainingData.get(item));
			int resultindex = (int) regressionModels.get(item).classifyInstance(newInstance);
			double[] distribution = regressionModels.get(item).distributionForInstance(newInstance);
			
			Enumeration<Object> attVals = trainingData.get(item).attribute(trainingData.get(item).classIndex()).enumerateValues();

			double[] values = Collections.list(attVals).stream().mapToDouble(o -> {
				return Double.parseDouble((String)o);
			}).toArray();
			
			double result = values[resultindex];
			System.out.println("values: " + Arrays.toString(values) + ", prediction: " + result + ", len: " + values.length);
			distributions.put(item,new double[][] {distribution,values});

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



	public RankClassificationRanker(String name) {
		super(name);
	}

	public ConflictResolutionStrategy getConflictResolutionStrategy() {
		return conflictResolutionStrategy;
	}

	public void setConflictResolutionStrategy(ConflictResolutionStrategy conflictResolutionStrategy) {
		this.conflictResolutionStrategy = conflictResolutionStrategy;
	}

}
