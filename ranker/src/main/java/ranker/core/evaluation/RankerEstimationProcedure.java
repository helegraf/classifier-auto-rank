package ranker.core.evaluation;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ranker.core.algorithms.PerfectRanker;
import ranker.core.algorithms.Ranker;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public abstract class RankerEstimationProcedure {

	protected StringBuilder summaryString = new StringBuilder();

	protected Ranker perfectRanker = new PerfectRanker();

	protected Map<RankerEvaluationMeasure,List<Double>> detailedEvaluationResults = new HashMap<RankerEvaluationMeasure,List<Double>>();
	protected List<List<Classifier>> predictedRankings = new ArrayList<List<Classifier>>();
	protected List<List<Classifier>> actualRankings = new ArrayList<List<Classifier>>();
	protected List<List<Double>> estimates = new ArrayList<List<Double>>();
	protected List<List<Double>> performanceValues = new ArrayList<List<Double>>();

	/**
	 * Has to divide data given into chunks of train/test splits & evaluate those on the evaluationProcedures
	 * 
	 * @param ranker
	 * @param evaluationProcedures
	 * @param data
	 * @param targetAttributes
	 * @return
	 * @throws Exception
	 */
	public abstract List<Double> estimate(Ranker ranker, List<RankerEvaluationMeasure> evaluationProcedures, Instances data,
			List<Integer> targetAttributes) throws Exception;

	public Map<RankerEvaluationMeasure,List<Double>> getDetailedEvaluationResults() {
		return detailedEvaluationResults;
	}

	public List<List<Classifier>> getPredictedRankings() {
		return predictedRankings;
	}

	public List<List<Classifier>> getActualRankings() {
		return actualRankings;
	}

	public List<List<Double>> getEstimates() {
		return estimates;
	}

	public List<List<Double>> getPerformanceValues() {
		return performanceValues;
	}

	public String getSummary() {
		return summaryString.toString();
	}

	/**
	 * Trains ranker on train set and evaluates it on each instance of the test set
	 * for each evaluation measure
	 * 
	 * @param ranker
	 * @param train
	 * @param test
	 * @param measures
	 * @param targetAttributes
	 * @return
	 * @throws Exception
	 *             If either the given ranker or the perfect ranker cannot be built
	 */
	protected void evaluateChunk(Ranker ranker, Instances train, Instances test,
			List<RankerEvaluationMeasure> measures, List<Integer> targetAttributes) throws Exception {
		// Build rankers
		ranker.buildRanker(train, targetAttributes);
		perfectRanker.buildRanker(train, targetAttributes);

		// Evaluate on each instance and get result from evaluationMeasure
		for (Instance instance : test) {
			try {
				// Build, get results and add to lists
				List<Classifier> predictedRanking = ranker.predictRankingforInstance(instance);
				List<Classifier> perfectRanking = perfectRanker.predictRankingforInstance(instance);
				List<Double> estimatedValues = ranker.getEstimates();
				List<Double> performanceMeasures = perfectRanker.getEstimates();
				predictedRankings.add(predictedRanking);
				actualRankings.add(perfectRanking);
				estimates.add(estimatedValues);
				performanceValues.add(performanceMeasures);

				// Evaluate on each measure & save results
				for (RankerEvaluationMeasure measure : measures) {
					double result = measure.evaluate(predictedRanking, perfectRanking, estimatedValues, performanceMeasures);
					detailedEvaluationResults.get(measure).add(result);
				}
			} catch (Exception e) {
				throw(e);
			}
		}
	}
}
