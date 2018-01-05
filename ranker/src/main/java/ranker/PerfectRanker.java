package ranker;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import de.upb.cs.is.jpl.api.dataset.defaultdataset.relative.Ranking;
import de.upb.cs.is.jpl.api.exception.algorithm.PredictionFailedException;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public class PerfectRanker extends PreferenceRanker {

	@Override
	public void buildRanker(Instances data) throws Exception {
		getClassifiersAndMetaFeatures(data);
		labels = new ArrayList<Integer>();
		indicesMap = new HashMap<Integer, Integer>();
		reverseIndicesMap = new HashMap<Integer, Integer>();
		getClassifiersAndMetaFeatures(data);
		for (int i = 0; i < classifierIndices.size(); i++) {
			labels.add(i);
			indicesMap.put(classifierIndices.get(i), i);
			reverseIndicesMap.put(i, classifierIndices.get(i));
		}
	}
	
	@Override
	public List<Classifier> predictRankingforInstance(Instance instance) throws PredictionFailedException {
		Ranking ranking = getRankingForInstance(instance);
		int[] rankingResult = ranking.getObjectList();
		List<Classifier> result = new ArrayList<Classifier>();
		for (int classifier : rankingResult) {
			result.add(classifiersMap.get(reverseIndicesMap.get(classifier)));
		}
		return result;
	}

}
