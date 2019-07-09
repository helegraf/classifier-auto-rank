package ranker.core.evaluation;

import java.util.ArrayList;
import java.util.List;

import jaicore.ml.weka.dataset.splitter.ArbitrarySplitter;
import ranker.Util;
import ranker.core.algorithms.Ranker;
import ranker.util.CSVHelper;
import weka.core.Instance;
import weka.core.Instances;

public class MCCV extends RankerEstimationProcedure {
	
	private double numbers;
	private double portions;
	private String suffix;
	
	public MCCV(double numbers, double portions, String suffix) {
		this.numbers = numbers;
		this.portions = portions;
		this.suffix = suffix;
	}

	@Override
	public List<Double> estimate(Ranker ranker, List<RankerEvaluationMeasure> measures, Instances data,
			List<Integer> targetAttributes) throws Exception {		
		
		List<String> measureNames = new ArrayList<>();
		
		// Initialize variables
		for (RankerEvaluationMeasure measure : measures) {
			detailedEvaluationResults.put(measure.getName(), new ArrayList<>());
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
		for (int i = 0; i < numbers; i++) {
			List<Instances> arbitrarySplit = new ArbitrarySplitter().split(data, i, portions);
			Instances train = new Instances(arbitrarySplit.get(0));
			Instances test = new Instances(arbitrarySplit.get(1));
			
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
		}
		
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
