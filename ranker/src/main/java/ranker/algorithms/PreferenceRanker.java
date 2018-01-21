package ranker.algorithms;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.TreeMap;

import de.upb.cs.is.jpl.api.algorithm.learningalgorithm.labelranking.ALabelRankingLearningModel;
import de.upb.cs.is.jpl.api.dataset.defaultdataset.relative.Ranking;
import de.upb.cs.is.jpl.api.dataset.labelranking.LabelRankingDataset;
import de.upb.cs.is.jpl.api.dataset.labelranking.LabelRankingInstance;
import de.upb.cs.is.jpl.api.exception.algorithm.PredictionFailedException;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public abstract class PreferenceRanker extends Ranker {

	/**
	 * Label ranking learning Model to predict the ranking
	 */
	ALabelRankingLearningModel learningModel;
	/**
	 * Contains a label for each classifier (cound from 0 -> n)
	 */
	ArrayList<Integer> labels;
	/**
	 * Maps classifier indices to labels
	 */
	HashMap<Integer, Integer> indicesMap;

	/**
	 * Maps labels to classifier indices
	 */
	HashMap<Integer, Integer> reverseIndicesMap;

	@Override
	public List<Classifier> predictRankingforInstance(Instance instance) throws PredictionFailedException {
		LabelRankingInstance labelRankingInstance = new LabelRankingInstance();
		labelRankingInstance.setRating(null);
		labelRankingInstance.setContextFeatureVector(getFeatureValuesForInstance(instance));
		labelRankingInstance.setContextId(0);
		labelRankingInstance.setTotalNumberOfLabels(labels.size());
		Ranking ranking = learningModel.predict(labelRankingInstance);
		int[] rankingResult = ranking.getObjectList();
		List<Classifier> result = new ArrayList<Classifier>();
		for (int classifier : rankingResult) {
			result.add(classifiersMap.get(reverseIndicesMap.get(classifier)));
		}
		int[] stuff = ranking.getOrderingForRanking();
		return result;
	}

	public LabelRankingDataset convertToLabelRankingDataSet(Instances data) throws Exception {
		ArrayList<double[]> featureValues = new ArrayList<double[]>();
		ArrayList<Ranking> rankings = new ArrayList<Ranking>();
		labels = new ArrayList<Integer>();
		indicesMap = new HashMap<Integer, Integer>();
		reverseIndicesMap = new HashMap<Integer, Integer>();
		
		// Ensure no missing vaules
		// TODO extract method
		ReplaceMissingValues filter = new ReplaceMissingValues();
		filter.setInputFormat(data);
		for (int i = 0; i < data.numInstances(); i++) {
			filter.input(data.instance(i));
		}
		filter.batchFinished();
		Instances newData = filter.getOutputFormat();
		Instance processed;
		while  ((processed=filter.output())!=null) {
			newData.add(processed);
		}

		getClassifiersAndMetaFeatures(newData);
		
		for (int i = 0; i < classifierIndices.size(); i++) {
			labels.add(i);
			indicesMap.put(classifierIndices.get(i), i);
			reverseIndicesMap.put(i, classifierIndices.get(i));
		}

		// Convert data given to ranking information
		newData.forEach(instance -> {
			// Feature information
			double[] featureValuesForInstance = getFeatureValuesForInstance(instance);
			featureValues.add(featureValuesForInstance);

			// Ranking
			Ranking result = getRankingForInstance(instance);

			rankings.add(result);
		});

		// Create LabelRankingDataset
		LabelRankingDataset dataset = new LabelRankingDataset(labels, featureValues, rankings);
		return dataset;
	}

	Ranking getRankingForInstance(Instance instance) {
		int[] objectList = new int[classifierIndices.size()];
		int[] compareOperators = new int[objectList.length - 1];
		for (int i = 0; i < compareOperators.length; i++) {
			compareOperators[i] = Ranking.COMPARABLE_ENCODING;
		}

		// For the instance, save different performance values together with classifiers
		// that have this value
		TreeMap<Double, ArrayList<Integer>> classifierPerformances = new TreeMap<Double, ArrayList<Integer>>();
		for (int attributeIndex : classifierIndices) {
			double attributeValue = instance.value(attributeIndex);
			if (classifierPerformances.containsKey(attributeValue)) {
				classifierPerformances.get(attributeValue).add(indicesMap.get(attributeIndex));
			} else {
				ArrayList<Integer> indices = new ArrayList<Integer>();
				indices.add(indicesMap.get(attributeIndex));
				classifierPerformances.put(attributeValue, indices);
			}
		}

		ArrayList<Integer> ordering = new ArrayList<Integer>();
		for (double performanceValue : classifierPerformances.descendingKeySet()) {
			ordering.addAll(classifierPerformances.get(performanceValue));
		}

		// For giving ranking
		// List<Integer> ranking = getRankingFromOrdering(ordering);
		// for (int i = 0; i < ranking.size(); i++) {
		// objectList[i] = ranking.get(i);
		// }

		// For giving ordering
		List<Integer> ranking = ordering;
		for (int i = 0; i < ranking.size(); i++) {
			objectList[i] = ranking.get(i);
		}

		Ranking result = new Ranking(objectList, compareOperators);
		return result;
	}

	List<Integer> getRankingFromOrdering(List<Integer> ordering) {
		Integer[] ranking = new Integer[ordering.size()];
		for (int i = 0; i < ordering.size(); i++) {
			ranking[ordering.get(i)] = i;
		}
		return Arrays.asList(ranking);
	}
}
