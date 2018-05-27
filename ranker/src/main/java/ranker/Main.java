package ranker;

import java.io.IOException;

import ranker.core.evaluation.MannWhitneyUEvaluator;
import ranker.util.ColumnNotFoundException;

public class Main {

	public static void main(String[] args) throws Exception {

		// DataSource source = new
		// DataSource("metaData_small_allPerformanceValues.arff");
		// Instances data = source.getDataSet();
		// data.deleteAttributeAt(0);
		//
		// ArrayList<Integer> targetAttributes = new ArrayList<Integer>();
		// for (int i = 103; i < 125; i++) {
		// targetAttributes.add(i);
		// }
		//
		// RandomForestRanker ranker = new RandomForestRanker();
		// ranker.buildRanker(data, targetAttributes);

		// Rankprediction pre = new Rankprediction();
		//
		// DataSource source = new
		// DataSource("src/main/resources/dataset_31_credit-g.arff");
		// Instances data = source.getDataSet();
		// data.setClassIndex(data.attribute("class").index());
		//
		// List<Classifier> classifs = pre.predictRanking(data);
		// classifs.forEach(classif ->
		// System.out.println(classif.getClass().getSimpleName()));

		// Read data set
		// DataSource source = new
		// DataSource("metaData_small_allPerformanceValues_onlyProbing.arff");
		// Instances data = source.getDataSet();
		// List<Integer> targetAttributes = new ArrayList<Integer>();
		// for (int i = 45; i < 67; i++) {
		// targetAttributes.add(i);
		// }
		//
		// Evaluate ranker
		// List<Double> evaluationResults =
		// EvaluationHelper.evaluateRegressionRanker(new RandomForestRanker(), data,
		// targetAttributes);
		// evaluationResults.forEach(result -> System.out.println(result));

		// evaluationResults = EvaluationHelper.evaluateRanker(new
		// LinearRegressionRanker(), data, targetAttributes);
		// evaluationResults.forEach(result -> System.out.println(result));
		//
		// evaluationResults = EvaluationHelper.evaluateRanker(new REPTreeRanker(),
		// data, targetAttributes);
		// evaluationResults.forEach(result -> System.out.println(result));
		//
		// evaluationResults = EvaluationHelper.evaluateRanker(new M5PRanker(), data,
		// targetAttributes);
		// evaluationResults.forEach(result -> System.out.println(result));

		// List<Double> evaluationResults = EvaluationHelper.evaluateRanker(new
		// PairwiseComparisonRanker(), data, targetAttributes);
		// evaluationResults.forEach(result -> System.out.println(result));
		//
		// evaluationResults = EvaluationHelper.evaluateRanker(new
		// InstanceBasedLabelRankingRanker(), data, targetAttributes);
		// evaluationResults.forEach(result -> System.out.println(result));
		//
		// evaluationResults = EvaluationHelper.evaluateRanker(new
		// InstanceBasedLabelRankingKemenyYoung(), data, targetAttributes);
		// evaluationResults.forEach(result -> System.out.println(result));
		//
		// evaluationResults = EvaluationHelper.evaluateRanker(new
		// InstanceBasedLabelRankingKemenyYoungSQRTN(), data, targetAttributes);
		// evaluationResults.forEach(result -> System.out.println(result));

		MannWhitneyUEvaluator.ALL_RANKER_NAMES.forEach(ranker -> {
			MannWhitneyUEvaluator.ALL_MEASURE_NAMES.forEach(measure -> {
				try {
					System.out.println(MannWhitneyUEvaluator.computeWhitneyU(ranker, MannWhitneyUEvaluator.NO_PROBING_META_DATA_DATA_SET_NAME,
							ranker, MannWhitneyUEvaluator.ALL_META_DATA_DATA_SET_NAME, measure));
				} catch (IOException | ColumnNotFoundException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			});
		});
		
		// TODO somehow save this.
	}

}
