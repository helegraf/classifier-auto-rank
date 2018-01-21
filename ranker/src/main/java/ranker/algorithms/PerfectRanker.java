package ranker.algorithms;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import de.upb.cs.is.jpl.api.dataset.defaultdataset.relative.Ranking;
import de.upb.cs.is.jpl.api.exception.algorithm.PredictionFailedException;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Ranker used for evaluation purposes. Has to be initialized with the test set,
 * which has to include performance values for the learning algorithms. The
 * ranker can then be queried for instances in the test set, and only those, and
 * returns the actual ranking of learning algorithms as induced from the
 * performance values.
 * 
 * @author Helena Graf
 *
 */
public class PerfectRanker extends PreferenceRanker {

	Instances data;

	@Override
	public void buildRanker(Instances data) throws Exception {
		getClassifiersAndMetaFeatures(data);
		labels = new ArrayList<Integer>();
		indicesMap = new HashMap<Integer, Integer>();
		reverseIndicesMap = new HashMap<Integer, Integer>();
		for (int i = 0; i < classifierIndices.size(); i++) {
			labels.add(i);
			indicesMap.put(classifierIndices.get(i), i);
			reverseIndicesMap.put(i, classifierIndices.get(i));
		}

		this.data = data;
	}

	@Override
	public List<Classifier> predictRankingforInstance(Instance instance) throws PredictionFailedException {
		// Search for the instance in the data
		
		Ranking ranking = getRankingForInstance(instance);
		int[] rankingResult = ranking.getObjectList();
		List<Classifier> result = new ArrayList<Classifier>();
		for (int classifier : rankingResult) {
			result.add(classifiersMap.get(reverseIndicesMap.get(classifier)));
		}
		return result;
	}

}
