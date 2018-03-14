package ranker.core.algorithms.preference;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.TreeMap;

import de.upb.cs.is.jpl.api.algorithm.learningalgorithm.labelranking.ALabelRankingLearningModel;
import de.upb.cs.is.jpl.api.dataset.defaultdataset.relative.Ranking;
import de.upb.cs.is.jpl.api.dataset.labelranking.LabelRankingDataset;
import de.upb.cs.is.jpl.api.dataset.labelranking.LabelRankingInstance;
import de.upb.cs.is.jpl.api.exception.algorithm.PredictionFailedException;
import ranker.core.algorithms.Ranker;
import ranker.util.wekaUtil.WEKAHelper;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public abstract class PreferenceRanker extends Ranker {

	/**
	 * Label ranking learning Model to predict the ranking
	 */
	protected ALabelRankingLearningModel learningModel;

	/**
	 * Contains a label for each learning algorithm (0 -> n-1 for n learning
	 * algorithms)
	 */
	protected ArrayList<Integer> labels;

	/**
	 * Maps classifier indices to labels
	 */
	protected HashMap<Integer, Integer> indicesMap;

	/**
	 * Maps labels to classifier indices
	 */
	protected HashMap<Integer, Integer> reverseIndicesMap;

	@Override
	public List<Classifier> predictRankingforInstance(Instance instance) throws PredictionFailedException {
		// Construct query instance
		LabelRankingInstance labelRankingInstance = new LabelRankingInstance();
		labelRankingInstance.setRating(null);
		labelRankingInstance.setContextFeatureVector(getFeatureValuesForInstance(instance));
		labelRankingInstance.setContextId(0);
		labelRankingInstance.setTotalNumberOfLabels(labels.size());

		// Map result
		Ranking ranking = learningModel.predict(labelRankingInstance);
		List<Classifier> result = mapToOrdering(ranking);
		return result;
	}
	
	@Override
	public List<Double> getEstimates() {
		return null;
		//throw new UnsupportedOperationException();
	}

	/**
	 * Initialize labels and mappings for labels
	 */
	protected void initializeLabels() {
		labels = new ArrayList<Integer>();
		indicesMap = new HashMap<Integer, Integer>();
		reverseIndicesMap = new HashMap<Integer, Integer>();
		for (int i = 0; i < targetAttributes.size(); i++) {
			labels.add(i);
			indicesMap.put(targetAttributes.get(i), i);
			reverseIndicesMap.put(i, targetAttributes.get(i));
		}
	}

	/**
	 * Map labels to corresponding learning algorithms.
	 * 
	 * @param ranking
	 *            The ranking of labels
	 * @return The ordering of learning algorithms
	 */
	protected List<Classifier> mapToOrdering(Ranking ranking) {
		int[] rankingResult = ranking.getObjectList();
		List<Classifier> result = new ArrayList<Classifier>();
		for (int classifier : rankingResult) {
			result.add(classifiersMap.get(reverseIndicesMap.get(classifier)));
		}
		return result;
	}

	protected LabelRankingDataset convertToLabelRankingDataSet(Instances data) throws Exception {
		// Replace missing values as preference algorithm can't deal with those
		Instances newData = WEKAHelper.replaceMissingValues(data);

		initializeLabels();

		// Convert data given to ranking information for each instance
		ArrayList<double[]> featureValues = new ArrayList<double[]>();
		ArrayList<Ranking> rankings = new ArrayList<Ranking>();
		newData.forEach(instance -> {
			featureValues.add(getFeatureValuesForInstance(instance));
			rankings.add(getRankingForInstance(instance));
		});

		// Merge to LabelRankingDataset
		LabelRankingDataset dataset = new LabelRankingDataset(labels, featureValues, rankings);
		return dataset;
	}

	/**
	 * Converts the performance measure information given for an instance to a
	 * ranking.
	 * 
	 * @param instance
	 * @return
	 */
	protected Ranking getRankingForInstance(Instance instance) {
		// Initialize temporary variables
		int[] objectList = new int[targetAttributes.size()];
		int[] compareOperators = new int[objectList.length - 1];
		for (int i = 0; i < compareOperators.length; i++) {
			compareOperators[i] = Ranking.COMPARABLE_ENCODING;
		}

		// Save performance values together with the learning algorithms that have
		// achieved this value
		TreeMap<Double, ArrayList<Integer>> classifierPerformances = new TreeMap<Double, ArrayList<Integer>>();
		for (int attributeIndex : targetAttributes) {
			double attributeValue = instance.value(attributeIndex);
			if (classifierPerformances.containsKey(attributeValue)) {
				classifierPerformances.get(attributeValue).add(indicesMap.get(attributeIndex));
			} else {
				ArrayList<Integer> indices = new ArrayList<Integer>();
				indices.add(indicesMap.get(attributeIndex));
				classifierPerformances.put(attributeValue, indices);
			}
		}

		// Ordering of learning algorithms
		ArrayList<Integer> ordering = new ArrayList<Integer>();
		for (double performanceValue : classifierPerformances.descendingKeySet()) {
			ordering.addAll(classifierPerformances.get(performanceValue));
		}
		for (int i = 0; i < ordering.size(); i++) {
			objectList[i] = ordering.get(i);
		}

		// Construct result
		Ranking result = new Ranking(objectList, compareOperators);
		return result;
	}
}
