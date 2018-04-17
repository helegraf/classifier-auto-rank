package ranker;

import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import ranker.core.algorithms.BestAlgorithmRanker;
import ranker.core.algorithms.preference.InstanceBasedLabelRankingKemenyYoung;
import ranker.core.algorithms.preference.InstanceBasedLabelRankingKemenyYoungSQRTN;
import ranker.core.algorithms.preference.InstanceBasedLabelRankingRanker;
import ranker.core.algorithms.preference.PairwiseComparisonRanker;
import ranker.core.algorithms.regression.LinearRegressionRanker;
import ranker.core.algorithms.regression.M5PRanker;
import ranker.core.algorithms.regression.REPTreeRanker;
import ranker.core.algorithms.regression.RandomForestRanker;
import ranker.core.evaluation.BestThreeLoss;
import ranker.core.evaluation.EvaluationHelper;
import ranker.core.evaluation.KendallRankCorrelation;
import ranker.core.evaluation.Loss;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Main {

	public static void main(String[] args) throws Exception {
		
//		DataSource source = new DataSource("metaData_small_allPerformanceValues.arff");
//		Instances data = source.getDataSet();
//		data.deleteAttributeAt(0);
//		
//		ArrayList<Integer> targetAttributes = new ArrayList<Integer>();
//		for (int i = 103; i < 125; i++) {
//			targetAttributes.add(i);
//		}
//		
//		RandomForestRanker ranker = new RandomForestRanker();
//		ranker.buildRanker(data, targetAttributes);
		
		Rankprediction pre = new Rankprediction();

		DataSource source = new DataSource("src/main/resources/dataset_31_credit-g.arff");
		Instances data = source.getDataSet();
		data.setClassIndex(data.attribute("class").index());
		
		List<Classifier> classifs = pre.predictRanking(data);
		classifs.forEach(classif-> 
		System.out.println(classif.getClass().getSimpleName()));
		
	}

	public static void evaluateAllRankersOnAllDataSets() throws IOException {
		String dataset = "metaData_small_allPerformanceValues.csv";
		evaluateAllRankersOnDataSet(dataset);

		dataset = "metaData_small_allPerformanceValues-weka.filters.unsupervised.attribute.Remove-R49-93.csv";
		evaluateAllRankersOnDataSet(dataset);
	}

	public static void evaluateAllRankersOnDataSet(String dataset) throws IOException {
		String[] algorithms = { InstanceBasedLabelRankingRanker.class.getSimpleName(),
				InstanceBasedLabelRankingKemenyYoung.class.getSimpleName(),
				InstanceBasedLabelRankingKemenyYoungSQRTN.class.getSimpleName(),
				PairwiseComparisonRanker.class.getSimpleName(), RandomForestRanker.class.getSimpleName(),
				LinearRegressionRanker.class.getSimpleName(), REPTreeRanker.class.getSimpleName(),
				M5PRanker.class.getSimpleName()};
		
		for (String firstAlgo : algorithms) {
			printAllMeasuresForAlgorithm(firstAlgo, dataset);
		}
	}

	public static void printAllMeasuresForAlgorithm(String firstAlgo, String dataset) throws IOException {
		String secondAlgo = BestAlgorithmRanker.class.getSimpleName();
		String[] measures = { KendallRankCorrelation.class.getSimpleName(), Loss.class.getSimpleName(),
				BestThreeLoss.class.getSimpleName() };

		for (String measure : measures) {
			printMeasureForAlgo(firstAlgo, secondAlgo, measure, dataset);
		}
	}

	public static void printMeasureForAlgo(String firstAlgo, String secondAlgo, String measure, String dataset)
			throws IOException {
		System.out.println(firstAlgo + " vs " + secondAlgo + " on " + dataset + " regarding " + measure);
		Path firstFile = FileSystems.getDefault().getPath(firstAlgo + "_" + dataset);
		Path secondFile = FileSystems.getDefault().getPath(secondAlgo + "_" + dataset);
		System.out.println("& " + EvaluationHelper.computeWhitneyU(firstFile, secondFile, measure));
	}
}
