package ranker.core.algorithms;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.TreeMap;

import weka.classifiers.Classifier;
import weka.core.Instance;

/**
 * Ranker used for evaluation purposes. Has to be built with
 * {@link #buildRanker(weka.core.Instances, List)} to know which attributes are
 * targets and which are meta features. Query instance given to
 * {@link #predictRankingforInstance(Instance)} has to include performance
 * values for the learning algorithm learning algorithms in order to return a
 * correct ranking. The method {@link #getEstimates()} then likewise returns the
 * correct performance values ordered according to the last returned ranking.
 * 
 * @author Helena Graf
 *
 */
public class PerfectRanker extends Ranker {

	/**
	 * Contains a label for each learning algorithm (0 to n-1 for n learning
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


	/**
	 * List with actual performance values ordered according to the last returned
	 * ranking. Null if no ranking has been predicted.
	 */
	protected List<Double> performanceMeasures;

	@Override
	public List<Classifier> predictRankingforInstance(Instance instance)  {
		Ranking ranking = getOrderingForInstance(instance);

		// Get actual performance measures
		performanceMeasures = new ArrayList<Double>();
		for (int label : ranking.getObjectList()) {
			int algorithmIndex = reverseIndicesMap.get(label);
			performanceMeasures.add(instance.value(algorithmIndex));
		}

		return mapOrderingToClassifiersList(ranking);
	}
	
	/**
	 * Map labels to corresponding learning algorithms.
	 * 
	 * @param ordering
	 *            The ranking of labels
	 * @return The ordering of learning algorithms
	 */
	protected List<Classifier> mapOrderingToClassifiersList(Ranking ordering) {
		int[] rankingResult = ordering.getObjectList();
		List<Classifier> result = new ArrayList<Classifier>();
		for (int classifier : rankingResult) {
			result.add(classifiersMap.get(reverseIndicesMap.get(classifier)));
		}
		return result;
	}
	
	/**
	 * Converts the performance measure information given for an instance to an
	 * ordering of labels.
	 * 
	 * @param instance
	 * @return
	 */
	protected Ranking getOrderingForInstance(Instance instance) {
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

	@Override
	public List<Double> getEstimates() {
		return performanceMeasures;
	}
	
	@Override
	protected void initialize() throws Exception {
		initializeLabels();
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
}
