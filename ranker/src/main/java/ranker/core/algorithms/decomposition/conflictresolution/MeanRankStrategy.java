package ranker.core.algorithms.decomposition.conflictresolution;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.TreeMap;

/**
 * Resolves conflicts in given ranking by ranking classifiers according to the
 * mean of the probabilities of predicted ranks multiplied by the ranks.
 * 
 * @author helegraf
 *
 */
public class MeanRankStrategy implements ConflictResolutionStrategy {

	@Override
	public void resolveConflictsAmongPredictions(TreeMap<Double, List<String>> predictions,
			HashMap<String, double[][]> distributions) {
		// just clear the previous predictions
		predictions.clear();

		// find value for each distribution
		distributions.forEach((item, distribution) -> {
			double value = 0;
			System.out.println("Dist len: " + distribution[0].length + "/" + distribution[1].length);
			for (int i = 0; i < distribution[0].length; i++) {
				value += distribution[0][i] * distribution[1][i];
			}
			value /= distribution[0].length;

			System.out.println("adding value: " + value + " + classifier: " + item);

			if (predictions.get(value) == null) {
				predictions.put(value, new ArrayList<>());
			}

			predictions.get(value).add(item);
		});

		System.out.println(predictions.size());
	}
}
