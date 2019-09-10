package ranker.core.algorithms.decomposition.conflictresolution;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.TreeMap;

/**
 * Attempts to resolve conflicts in a ranking in the following way:
 * 
 * <ol>
 * <li>Starting from the first rank, check if the same rank is predicted for two
 * algorithms. In this case, only keep this predicted rank for the algorithm for
 * which this rank has the highest probability.</li>
 * 
 * <li>All other algorithms are assigned the next rank that has the highest
 * probability (except ones that have already been tried in the course of the
 * conflict resolution).</li>
 * 
 * <li>If all possible ranks have been tried for an algorithm, it is queued. At
 * the end, the queued list is assigned the previously unsassigned ranks,
 * starting from the first rank and first queue element.</li>
 * </ol>
 * 
 * @author helegraf
 *
 */
public class HighestProbabilityStrategy implements ConflictResolutionStrategy {

	@Override
	public void resolveConflictsAmongPredictions(TreeMap<Double, List<String>> predictions,
			HashMap<String, double[][]> distributions) {

		HashMap<String, double[][]> newDistributions = new HashMap<>();
		distributions.forEach((item, distribution) -> {
			newDistributions.put(item, Arrays.copyOf(distribution, distribution.length));
		});

		List<String> unassignedClassifiers = new ArrayList<>();
		HashMap<String, Double> oldPredictions = new HashMap<>();
		HashMap<String, Double> newPredictions = new HashMap<>();

		do {
			oldPredictions.clear();
			newPredictions.clear();

			// find conflicts
			predictions.descendingMap().forEach((value, classifierList) -> {
				if (classifierList != null && classifierList.size() > 1) {
					// remove all except the one with the highest confidence for the value (first
					// one best)
					double confidence = 0;
					String best = null;

					// find out best
					for (String classifier : classifierList) {
						double newConfidence = getConfidenceForClass(classifier, value, newDistributions);
						if (newConfidence > confidence) {
							// move the previous one away
							if (best != null) {
								assignNextPriority(best, value, newDistributions.get(best), newPredictions,
										oldPredictions, unassignedClassifiers);
							}

							confidence = newConfidence;
							best = classifier;
						} else {
							// move the current one away
							assignNextPriority(classifier, value, newDistributions.get(classifier), newPredictions,
									oldPredictions, unassignedClassifiers);
						}
					}
				}
			});

			// remove conflicting
			oldPredictions.forEach((classifier, classVal) -> {
				predictions.get(classVal).remove(classifier);
			});

			// add new
			newPredictions.forEach((classifier, classVal) -> {
				if (predictions.get(classVal) == null) {
					predictions.put(classVal, new ArrayList<>());
				}

				predictions.get(classVal).add(classifier);
			});

		} while (!newPredictions.isEmpty());

		// add leftovers
		for (double i = 0; i < predictions.size(); i++) {
			if (predictions.get(i) == null) {
				predictions.put(i, Arrays.asList(unassignedClassifiers.get(0)));
				unassignedClassifiers.remove(0);
			}
		}
	}

	private double getConfidenceForClass(String item, double clazz, HashMap<String, double[][]> distributions) {
		double[][] distributionsForClassifire = distributions.get(item);
		for (int i = 0; i < distributionsForClassifire[1].length; i++) {
			if (distributionsForClassifire[1][i] == clazz) {
				return distributionsForClassifire[0][i];
			}
		}

		throw new IllegalArgumentException();
	}

	private void assignNextPriority(String item, double conflictingClass, double[][] distributions,
			HashMap<String, Double> newPredictions, HashMap<String, Double> oldPredictions,
			List<String> unassignedItems) {
		// mark old class for removal
		oldPredictions.put(item, conflictingClass);

		// remove the probability that the classifier will be moved to this class again
		for (int i = 0; i < distributions[1].length; i++) {
			if (distributions[1][i] == conflictingClass) {
				distributions[0][i] = 0;
				break;
			}
		}

		// find max probability
		double max = 0;
		int index = -1;
		for (int i = 0; i < distributions[0].length; i++) {
			if (distributions[0][i] > max) {
				max = distributions[0][i];
				index = i;
			}
		}

		// assign new class
		if (max == 0) {
			unassignedItems.add(item);
		} else {
			newPredictions.put(item, distributions[1][index]);
		}

	}

}
