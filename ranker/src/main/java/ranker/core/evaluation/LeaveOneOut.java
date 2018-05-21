package ranker.core.evaluation;

import java.util.ArrayList;
import java.util.List;

import ranker.Util;
import ranker.core.algorithms.Ranker;
import ranker.util.CSVHelper;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Divides data into n different splits of n-1 instances as a train set and the
 * remaining instance as the test set. Averages the result of these n runs,
 * leaving out NaN values (these are still included in
 * {@link #getDetailedEvaluationResults()} thought).
 * 
 * @author Helena Graf
 *
 */
public class LeaveOneOut extends RankerEstimationProcedure {

	@Override
	public List<Double> estimate(Ranker ranker, List<RankerEvaluationMeasure> measures, Instances metaData,
			List<Integer> targetAttributes) throws Exception {
		// Initialize variables
		for (RankerEvaluationMeasure measure : measures) {
			detailedEvaluationResults.put(measure.getClass().getSimpleName(), new ArrayList<Double>());
		}
		detailedEvaluationResults.put(Util.DATA_ID, new ArrayList<>());
		detailedEvaluationResults.put(Util.RANKER_BUILD_TIMES, new ArrayList<>());
		detailedEvaluationResults.put(Util.RANKER_PREDICT_TIMES, new ArrayList<>());

		// Evaluate all instances separately
		for (int i = 0; i < metaData.numInstances(); i++) {
			Instances train = new Instances(metaData);
			Instance instanceToTest = train.remove(i);
			ArrayList<Attribute> attributes = new ArrayList<Attribute>();
			for (int j = 0; j < metaData.numAttributes(); j++) {
				attributes.add(metaData.attribute(j));
			}
			Instances test = new Instances("Test", attributes, 0);
			test.add(instanceToTest);
			detailedEvaluationResults.get(Util.DATA_ID).add((double) instanceToTest.value(0));
			// remove DataIds, we do not want them as values for testing
			train.deleteAttributeAt(0);
			test.deleteAttributeAt(0);
			List<Integer> decrementedTargetAttributes = new ArrayList<>();
			targetAttributes.forEach(number -> decrementedTargetAttributes.add(new Integer(number - 1)));
			evaluateChunk(ranker, train, test, measures, decrementedTargetAttributes);
		}
		//TODO maybe don't automatically write evaluation results?
		CSVHelper.writeCSVFile(CSVHelper.getCSVPath(ranker, metaData), detailedEvaluationResults);

		// Average results
		int numInstancesCalculated = metaData.numInstances();
		List<Double> results = new ArrayList<Double>();
		for (String measure : detailedEvaluationResults.keySet()) {
			double result = 0;
			for (double value : detailedEvaluationResults.get(measure)) {
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
