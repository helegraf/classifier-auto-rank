package ranker;

import java.util.ArrayList;
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

public abstract class PreferenceRanker extends Ranker {
	
	ALabelRankingLearningModel learningModel;
	ArrayList<Integer> labels;
	HashMap<Integer,Integer> indicesMap;
	
	@Override
	public List<Classifier> predictRankingforInstance(Instance instance) throws PredictionFailedException {
		LabelRankingInstance labelRankingInstance = new LabelRankingInstance();
		labelRankingInstance.setRating(null);
		labelRankingInstance.setContextFeatureVector(getFeatureValuesForInstance(instance));
		labelRankingInstance.setContextId(0);
		labelRankingInstance.setTotalNumberOfLabels(labels.size());
		Ranking ranking = learningModel.predict(labelRankingInstance);
		int [] rankingResult = ranking.getObjectList();
		List<Classifier> result = new ArrayList<Classifier>();
		for (int classifier : rankingResult) {
			result.add(classifiersMap.get(classifier));
		}
		return result;
	}
	
	public LabelRankingDataset convertToLabelRankingDataSet(Instances data) throws Exception {
		ArrayList<double[]> featureValues = new ArrayList<double[]>();
		ArrayList<Ranking> rankings = new ArrayList<Ranking>();
		labels = new ArrayList<Integer>();
		indicesMap = new HashMap<Integer,Integer>();
		
		getClassifiersAndMetaFeatures(data);
		for (int i = 0; i < classifierIndices.size(); i++) {
			labels.add(i);
			indicesMap.put(classifierIndices.get(i), i);
		}
				
		// Convert data given to ranking information
		data.forEach(instance->{
			// Feature information
			featureValues.add(getFeatureValuesForInstance(instance));
			
			// Ranking
			int[] objectList = new int[classifierIndices.size()];
			int[] compareOperators = new int[objectList.length-1];
			for (int i = 0; i < compareOperators.length; i++) {
				compareOperators[i] = Ranking.COMPARABLE_ENCODING;
			}
			
			TreeMap<Double,ArrayList<Integer>> classifierPerformances = new TreeMap<Double,ArrayList<Integer>>();
			for (int attributeIndex : classifierIndices) {
				double attributeValue = instance.value(attributeIndex);
				if (classifierPerformances.containsKey(attributeValue)) {
					classifierPerformances.get(attributeValue).add(attributeIndex);
				} else {
					ArrayList<Integer> indices = new ArrayList<Integer>();
					indices.add(attributeIndex);
					classifierPerformances.put(attributeValue, indices);
				}
			}
			
			int index = 0;
			for (double performanceValue : classifierPerformances.descendingKeySet()) {
				ArrayList<Integer> indices = classifierPerformances.get(performanceValue);
				if (indices.size() > 1) {
					for (int i = 0; i < indices.size(); i++) {
						objectList[index] = indicesMap.get(indices.get(i));
						// no equals of equal objects because this cannot be handled by framework
						index++;
					}
				} else {
					objectList[index] = indicesMap.get(indices.get(0));
					index++;
				}
			}
			
			Ranking ranking = new Ranking(objectList, compareOperators);
			rankings.add(ranking);
		});
		
		// Create LabelRankingDataset
		LabelRankingDataset dataset = new LabelRankingDataset(labels,featureValues,rankings);
		return dataset;
	}
}
