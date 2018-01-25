package rankerEvaluation;

import java.util.ArrayList;
import java.util.List;

import ranker.algorithms.Ranker;
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
	public List<Double> estimate(Ranker ranker, List<RankerEvaluationMeasure> measures, Instances data,
			List<Integer> targetAttributes) throws Exception {
		// Initialize variables
		for (RankerEvaluationMeasure measure : measures) {
			detailedEvaluationResults.put(measure, new ArrayList<Double>());
		}
		
		// Evaluate all instances separately
		for (int i = 0; i < data.numInstances(); i++) {
			Instances train = new Instances(data);
			Instance remove = train.remove(i);
			ArrayList<Attribute> attributes = new ArrayList<Attribute>();
			for (int attribute = 0; attribute < data.numAttributes(); attribute++) {
				attributes.add(data.attribute(attribute));
			}
			Instances test = new Instances("Test", attributes, 0);
			test.add(remove);

			evaluateChunk(ranker, train, test, measures, targetAttributes);
		}

		// Average results
		int numInstancesCalculated = data.numInstances();
		List<Double> results = new ArrayList<Double>();
		for (RankerEvaluationMeasure measure : detailedEvaluationResults.keySet()) {
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
