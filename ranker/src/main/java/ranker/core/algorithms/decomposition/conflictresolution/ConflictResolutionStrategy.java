package ranker.core.algorithms.decomposition.conflictresolution;

import java.util.HashMap;
import java.util.List;
import java.util.TreeMap;

public interface ConflictResolutionStrategy {

	/**
	 * Modifies the given predictions, attempting to resolve conflicts (several
	 * classifier for which the same class has been predicted). Does not necessarily
	 * resolve all conflicts.
	 * 
	 * @param predictions   the predicted class together with the classifiers for
	 *                      which it was predicted
	 * @param distributions the class distributions + class values for each
	 *                      classifier
	 */
	public void resolveConflictsAmongPredictions(TreeMap<Double, List<String>> predictions,
			HashMap<String, double[][]> distributions);
}
