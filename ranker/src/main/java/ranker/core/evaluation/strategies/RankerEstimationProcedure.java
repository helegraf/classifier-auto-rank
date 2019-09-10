package ranker.core.evaluation.strategies;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang3.time.StopWatch;

import ranker.Util;
import ranker.core.algorithms.Ranker;
import ranker.core.algorithms.baseline.PerfectRanker;
import ranker.core.evaluation.measures.RankerEvaluationMeasure;
import weka.core.Instance;
import weka.core.Instances;

public abstract class RankerEstimationProcedure {

	protected StringBuilder summaryString = new StringBuilder();

	protected Ranker perfectRanker = new PerfectRanker();

	protected Map<String, List<Object>> detailedEvaluationResults = new HashMap<>();
	protected List<List<String>> predictedRankings = new ArrayList<>();
	protected List<List<String>> actualRankings = new ArrayList<>();
	protected List<List<Double>> estimates = new ArrayList<>();
	protected List<List<Double>> performanceValues = new ArrayList<>();
	private StopWatch watch = new StopWatch();

	/**
	 * Has to divide data given into chunks of train/test splits and evaluate those
	 * on the evaluationProcedures
	 * 
	 * @param ranker               the ranker for which to make an estimate
	 * @param evaluationProcedures the measures to use
	 * @param data                 the data on which to evaluate
	 * @param targetAttributes     the attributes of the data that contain algorithm
	 *                             performances
	 * @return estimates for all given measures
	 * @throws Exception if the evaluation cannot be completed
	 */
	public abstract List<Double> estimate(Ranker ranker, List<RankerEvaluationMeasure> evaluationProcedures,
			Instances data, List<Integer> targetAttributes) throws Exception;

	public Map<String, List<Object>> getDetailedEvaluationResults() {
		return detailedEvaluationResults;
	}

	public List<List<String>> getPredictedRankings() {
		return predictedRankings;
	}

	public List<List<String>> getActualRankings() {
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
	 * @param ranker           the ranker to evaluate
	 * @param train            the part of the data the ranker is trained on
	 * @param test             the part of the data the ranker is tested on
	 * @param measures         the measures to evaluate
	 * @param targetAttributes the attributes of the data that contain performance
	 *                         values of algoirthms
	 * @throws Exception If either the given ranker or the perfect ranker cannot be
	 *                   built
	 */
	protected void evaluateChunk(Ranker ranker, Instances train, Instances test, List<RankerEvaluationMeasure> measures,
			List<Integer> targetAttributes) throws Exception {
		// Build rankers

		// Stop times of ranking
		watch.reset();
		watch.start();
		ranker.buildRanker(train, targetAttributes);
		watch.stop();
		double buildTime = (double) watch.getTime();
		perfectRanker.buildRanker(train, targetAttributes);

		// Evaluate on each instance and get result from evaluationMeasure
		for (Instance instance : test) {
			try {
				// Build, get results and add to lists
				watch.reset();
				watch.start();
				List<String> predictedRanking = ranker.predictRankingforInstance(instance);
				watch.stop();
				double predictTime = (double) watch.getTime();
				detailedEvaluationResults.get(Util.RANKER_PREDICT_TIMES).add(predictTime);
				List<String> perfectRanking = perfectRanker.predictRankingforInstance(instance);
				List<Double> estimatedValues = ranker.getEstimates();
				List<Double> performanceMeasures = perfectRanker.getEstimates();
				predictedRankings.add(predictedRanking);
				actualRankings.add(perfectRanking);
				estimates.add(estimatedValues);
				performanceValues.add(performanceMeasures);

				// Evaluate on each measure & save results
				for (RankerEvaluationMeasure measure : measures) {
					double result = measure.evaluate(predictedRanking, perfectRanking, estimatedValues,
							performanceMeasures);
					detailedEvaluationResults.get(measure.getName()).add(result);
				}
				// save time of ranker building
				detailedEvaluationResults.get(Util.RANKER_BUILD_TIMES).add(buildTime);
				detailedEvaluationResults.get("classifier_string").add(ranker.getClassifierString());

				String predictedRankingString = "";
				for (String item : predictedRanking) {
					predictedRankingString += item + ", ";
				}

				String actualRankingString = "";
				for (int i = 0; i < perfectRanking.size(); i++) {
					actualRankingString += perfectRanking.get(i) + ", ";
				}

				detailedEvaluationResults.get("predicted_ranking").add(predictedRankingString);
				detailedEvaluationResults.get("actual_ranking").add(actualRankingString);
			} catch (Exception e) {
				throw (e);
			}
		}
	}

}
