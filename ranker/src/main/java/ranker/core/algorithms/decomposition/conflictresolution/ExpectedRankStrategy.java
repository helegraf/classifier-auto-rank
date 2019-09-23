package ranker.core.algorithms.decomposition.conflictresolution;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.TreeMap;

/**
 * Resolves conflicts in given ranking by ranking classifiers according to the
 * probabilities of predicted ranks multiplied by the ranks (the expected value
 * for the rank).
 * 
 * @author helegraf
 *
 */
public class ExpectedRankStrategy implements ConflictResolutionStrategy {

	@Override
	public void resolveConflictsAmongPredictions(TreeMap<Double, List<String>> predictions,
			HashMap<String, double[][]> distributions) {
		// just clear the previous predictions
		predictions.clear();

		// find value for each distribution
		distributions.forEach((item, distribution) -> {
			double value = 0;

			for (int i = 0; i < distribution[0].length; i++) {
				value += distribution[0][i] * distribution[1][i];
			}

			if (predictions.get(value) == null) {
				predictions.put(value, new ArrayList<>());
			}

			predictions.get(value).add(item);
		});
	}
}
