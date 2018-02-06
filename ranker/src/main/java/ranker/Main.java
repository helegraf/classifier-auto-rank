package ranker;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import ranker.core.algorithms.BestAlgorithmRanker;
import ranker.core.algorithms.InstanceBasedLabelRankingRanker;
import ranker.core.algorithms.PairwiseComparisonRanker;
import ranker.core.algorithms.PerfectRanker;
import ranker.core.algorithms.RandomForestRanker;
import ranker.core.algorithms.Ranker;
import ranker.core.algorithms.RegressionRanker;
import ranker.core.evaluation.EvaluationHelper;
import ranker.core.evaluation.KendallRankCorrelation;
import ranker.core.evaluation.LeaveOneOut;
import ranker.core.evaluation.Loss;
import ranker.core.evaluation.RankerEvaluationMeasure;
import ranker.core.metafeatures.GlobalCharacterizer;
import ranker.core.metafeatures.MetaFeatureHelper;
import ranker.util.openMLUtil.OpenMLHelper;
import ranker.util.wekaUtil.WekaHelper;
import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

public class Main {

	public static void main(String[] args) throws Exception {

//		List<Integer> datasets = OpenMLHelper.getDataSetsFromIndex();
//		// List<Integer> datasets = Arrays.asList(3,4,5);
//		Instances instances1 = MetaFeatureHelper.computeMetaFeatures(datasets);
//		ArffSaver saver = new ArffSaver();
//		saver.setInstances(instances1);
//		try {
//			saver.setFile(new File(instances1.relationName() + ".arff"));
//			saver.writeBatch();
//		} catch (IOException e) {
//			e.printStackTrace();
//		}

		// BufferedReader reader =
		// Files.newBufferedReader(FileSystems.getDefault().getPath("meta_computed.arff"),
		// Util.charset);
		// Instances instances = new Instances(reader);
		// RegressionRanker ranker = new RandomForestRanker();
		// //System.out.println(Util.testRanker(ranker, instances));
		// ranker.buildRanker(instances);
		// ranker.predictRankingforInstance(instances.get(8));
		// System.out.println(ranker.getEstimatesForRanking());
		// ranker.predictRankingforInstance(instances.get(200));
		// System.out.println(ranker.getEstimatesForRanking());
		// ranker.predictRankingforInstance(instances.get(388));
		// System.out.println(ranker.getEstimatesForRanking());

		// ArrayList<Integer> dataIds = new ArrayList<Integer>();
		// BufferedReader reader =
		// Files.newBufferedReader(FileSystems.getDefault().getPath("datasets_100_1000"));
		// String line = null;
		// while ((line=reader.readLine())!=null) {
		// dataIds.add(Integer.parseInt(line));
		// }
		// Instances results = Util.computeMetaFeatures(dataIds);
		// ArffSaver saver = new ArffSaver();
		// saver.setInstances(results);
		// try {
		// saver.setFile(new File("meta_computed.arff"));
		// saver.writeBatch();
		// } catch (IOException e) {
		// e.printStackTrace();
		// }

		// Trying to save a .csv
//		 Path path = FileSystems.getDefault().getPath("datatest.csv");
//		 BufferedWriter writer = Files.newBufferedWriter(path, Util.charset);
		//
		// writer.write("1");
		// writer.write(seperator);
		// writer.write("2");
		// writer.write(seperator);
		// writer.write("3");
		// writer.write(seperator);
		//
		// writer.newLine();
		//
		// writer.write("1");
		// writer.write(seperator);
		// writer.write("2");
		// writer.write(seperator);
		// writer.write("3");
		// writer.write(seperator);
		//
		// writer.flush();
		// writer.close();

//		BufferedReader reader = Files.newBufferedReader(FileSystems.getDefault().getPath("metaData_small_allPerformanceValues_noID.arff"),
//				Util.charset);
//
//		Instances instances = new Instances(reader);
//		BestAlgorithmRanker ranker = new BestAlgorithmRanker();
//		
//		List<Integer> targetAttributes = new ArrayList<Integer>();
//		for (int i = 103; i < 125; i++) {
//			targetAttributes.add(i);
//		}
//		
//		ranker.buildRanker(instances, targetAttributes);
//		List<Classifier> ranking = ranker.predictRankingforInstance(instances.get(0));
//		List<Integer> stats = ranker.getClassifierStats();
//		for (int i = 0; i < ranking.size(); i++) {
//			System.out.println(ranking.get(i).getClass().getSimpleName() + " " + stats.get(i));
//		}
//		
		//System.out.println(EvaluationHelper.evaluateRanker(ranker, WekaHelper.subSet(instances, 1, 50), targetAttributes));
		
		BufferedReader reader = Files.newBufferedReader(FileSystems.getDefault().getPath("metaData_small_allPerformanceValues.arff"),
				Util.charset);

		Instances instances = new Instances(reader);
		Ranker ranker = new RandomForestRanker();
		
		List<Integer> targetAttributes = new ArrayList<Integer>();
		for (int i = 104; i < 126; i++) {
			targetAttributes.add(i);
		}
		
		System.out.println(EvaluationHelper.evaluateRanker(ranker, WekaHelper.subSet(instances, 1, 4), targetAttributes));
	
		//System.out.println(EvaluationHelper.evaluateRanker(ranker, instances, targetAttributes));
//		
//		Rankprediction rankprediction = new Rankprediction(instances,targetAttributes,new GlobalCharacterizer(),ranker);
//		List<Classifier> classifs = rankprediction.predictRanking(OpenMLHelper.getInstancesById(2));
//		classifs.forEach(elem -> System.out.println(elem.getClass().getSimpleName()));
		
		
	}
}
