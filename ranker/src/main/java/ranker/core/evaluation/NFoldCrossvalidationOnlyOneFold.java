package ranker.core.evaluation;

import java.util.ArrayList;
import java.util.List;

import ranker.Util;
import ranker.core.algorithms.Ranker;
import ranker.util.CSVHelper;
import weka.core.Instance;
import weka.core.Instances;

public class NFoldCrossvalidationOnlyOneFold extends RankerEstimationProcedure {

	private int folds;
	private String suffix;
	private int fold;

	public NFoldCrossvalidationOnlyOneFold(int folds, int fold) {
		this(folds, fold, "");
	}

	public NFoldCrossvalidationOnlyOneFold(int folds, int fold, String suffix) {
		this.folds = folds;
		this.suffix = fold + "_" + suffix;
		this.fold = fold;
	}

	@Override
	public List<Double> estimate(Ranker ranker, List<RankerEvaluationMeasure> measures, Instances data,
			List<Integer> targetAttributes) throws Exception {

		List<String> measureNames = new ArrayList<>();

		// Initialize variables
		for (RankerEvaluationMeasure measure : measures) {
			detailedEvaluationResults.put(measure.getName(), new ArrayList<Object>());
			measureNames.add(measure.getName());
		}
		detailedEvaluationResults.put(Util.DATA_ID, new ArrayList<>());
		detailedEvaluationResults.put(Util.RANKER_BUILD_TIMES, new ArrayList<>());
		detailedEvaluationResults.put(Util.RANKER_PREDICT_TIMES, new ArrayList<>());
		detailedEvaluationResults.put("actual_ranking", new ArrayList<>());
		detailedEvaluationResults.put("predicted_ranking", new ArrayList<>());
		detailedEvaluationResults.put("fold_number", new ArrayList<>());
		detailedEvaluationResults.put("classifier_string", new ArrayList<>());

		// Evaluate all instances separately
		int i = fold;
		Instances train = data.trainCV(folds, i);
		Instances test = data.testCV(folds, i);

		for (Instance instanceToTest : test) {
			detailedEvaluationResults.get(Util.DATA_ID).add((double) instanceToTest.value(0));
			detailedEvaluationResults.get("fold_number").add(i);
		}

		// remove DataIds, we do not want them as values for testing
		train.deleteAttributeAt(0);
		test.deleteAttributeAt(0);
		List<Integer> decrementedTargetAttributes = new ArrayList<>();
		targetAttributes.forEach(number -> decrementedTargetAttributes.add(new Integer(number - 1)));
		evaluateChunk(ranker, train, test, measures, decrementedTargetAttributes);

		CSVHelper.writeCSVFile(CSVHelper.getCSVPath(ranker, data, suffix), detailedEvaluationResults);

		// Average results
		int numInstancesCalculated = 0;
		List<Double> results = new ArrayList<>();
		for (String measure : measureNames) {
			numInstancesCalculated = detailedEvaluationResults.get(measure).size();
			double result = 0;
			for (Object val : detailedEvaluationResults.get(measure)) {
				double value = (double) val;
				if (!Double.isNaN(value)) {
					result += value;
				} else {
					numInstancesCalculated--;
				}
			}
			result /= numInstancesCalculated;
			results.add(result);
		}

		return results;
	}
}
