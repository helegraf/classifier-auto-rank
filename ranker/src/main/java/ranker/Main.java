//package ranker;
//
//import java.io.BufferedReader;
//import java.io.IOException;
//import java.nio.file.FileSystems;
//import java.nio.file.Files;
//import java.sql.ResultSet;
//import java.sql.SQLException;
//import java.util.ArrayList;
//import java.util.Arrays;
//import java.util.HashMap;
//import java.util.List;
//
//import dataHandling.mySQL.CustomMySQLAdapter;
//import dataHandling.mySQL.MetaDataDataBaseConnection;
//import de.upb.cs.is.jpl.api.algorithm.baselearner.regression.logisticWEKA.LogisticRegressionWEKA;
//import de.upb.cs.is.jpl.api.algorithm.learningalgorithm.labelranking.labelrankingbypairwisecomparison.LabelRankingByPairwiseComparisonLearningAlgorithm;
//import de.upb.cs.is.jpl.api.algorithm.learningalgorithm.labelranking.labelrankingbypairwisecomparison.LabelRankingByPairwiseComparisonLearningModel;
//import de.upb.cs.is.jpl.api.dataset.defaultdataset.relative.Ranking;
//import de.upb.cs.is.jpl.api.dataset.labelranking.LabelRankingDataset;
//import de.upb.cs.is.jpl.api.exception.algorithm.PredictionFailedException;
//import de.upb.cs.is.jpl.api.exception.algorithm.TrainModelsFailedException;
//import de.upb.cs.is.jpl.api.math.RandomGenerator;
//import ranker.core.metafeatures.MetaFeatureHelper;
//import weka.classifiers.AbstractClassifier;
//import weka.core.Instances;
//import weka.core.OptionHandler;
//
//public class Main {
//
//	public static void main(String[] args) throws Exception {
//
//		MetaDataDataBaseConnection connect = new MetaDataDataBaseConnection("isys-db.cs.upb.de", "hgraf",
//				"wzq0A1GXAj9sxKVH", "hgraf");
//
////		BufferedReader readr = Files.newBufferedReader(
////				FileSystems.getDefault().getPath("src", "main", "resources", Util.META_DATA_SMALL_DATA_SETS_COMPUTED),
////				Util.CHARSET);
////		Instances instances = new Instances(readr);
////		HashMap<String, List<Double>> metainfo = readMetaFeatureGroupTimes();
////		metainfo.forEach((a, b) -> {
////			System.out.println(a + " " + b.size());
////		});
////		
////		for (int i = 0; i < instances.numInstances(); i++) {
////			HashMap<String, Double> featureValues = new HashMap<String, Double>();
////			
////			for (int j = 1; j < 104; j++) {
////				String attname = instances.get(i).attribute(j).name();
////				double attval = instances.get(i).value(j);
////				featureValues.put(attname, attval);
////			}
////			
////			HashMap<String, Double> groupTimes = new HashMap<String, Double>();
////			
////			for (String metafeaturegroupe : metainfo.keySet()) {
////				groupTimes.put(metafeaturegroupe, metainfo.get(metafeaturegroupe).get(i));
////			}
////			
////			connect.addMetaDataForDataSet((int) instances.get(i).value(0), featureValues, groupTimes, null);
////		}
//		
//		CustomMySQLAdapter adapter = new CustomMySQLAdapter("isys-db.cs.upb.de", "hgraf",
//				"wzq0A1GXAj9sxKVH", "hgraf");
//		ResultSet resultSet = adapter.getResultsOfQuery("SELECT * FROM dataset_ids WHERE dataset_id NOT IN (SELECT dataset_id FROM classifier_runs)");
//		List<Integer> missingDids = new ArrayList<Integer>();
//		while(resultSet.next()) {
//			missingDids.add(resultSet.getInt("dataset_id"));
//		}
//		adapter.close();
//		
//		System.out.println(missingDids.size());
//		System.out.println(missingDids);
//		Instances missingInfo = MetaFeatureHelper.gatherClassifierPerformanceValues(missingDids);
//		System.out.println(missingInfo);
//		
//		
//		
//		missingInfo.forEach(instance -> {
//			for (int i = 1; i < instance.numAttributes(); i++) {
//				String classifierWithConfig = null;
//				try {
//					classifierWithConfig = instance.attribute(i).name() + MetaDataDataBaseConnection.CLASSIFIER_NAME_CONFIG_SEPARATOR + Arrays.toString(((OptionHandler) (AbstractClassifier.forName(instance.attribute(i).name(), null))).getOptions());
//					System.out.println(classifierWithConfig);
//				} catch (Exception e1) {
//					// TODO Auto-generated catch block
//					e1.printStackTrace();
//				}
//				try {
//					connect.addClassifierPerformance((int)instance.value(0), classifierWithConfig, "10X-stratified-CCV", instance.value(i));
//				} catch (SQLException e) {
//					// TODO Auto-generated catch block
//					e.printStackTrace();
//				}
//			}
//			
//			
//		});
//		//
//		// --------------------------------------------------------------------------------------------------
//
//		// System.out.println(Util.META_DATA_SMALL_DATA_SETS_COMPUTED);
//		// InputStream inputStream =
//		// Thread.currentThread().getContextClassLoader().getResourceAsStream(Util.META_DATA_SMALL_DATA_SETS_COMPUTED);
//		// DataSource source = new DataSource(inputStream);
//		// Instances data = source.getDataSet();
//		//
//		// ArrayList<Integer> targetAttributes = new ArrayList<Integer>();
//		// for (int i = 104; i < 126; i++) {
//		// targetAttributes.add(i);
//		// }
//		//
//		// List<Double> evaluationResults = EvaluationHelper.evaluateRanker(new
//		// PairwiseComparisonWEKARanker(),
//		// data, targetAttributes);
//		// evaluationResults.forEach(result -> System.out.println(result));
//
//		// ---------------------------------------------------------------------------------------------------------
//		//
//		// RandomForestRanker ranker = new RandomForestRanker();
//		// ranker.buildRanker(data, targetAttributes);
//
//		// Rankprediction pre = new Rankprediction();
//		//
//		// DataSource source = new
//		// DataSource("src/main/resources/dataset_31_credit-g.arff");
//		// Instances data = source.getDataSet();
//		// data.setClassIndex(data.attribute("class").index());
//		//
//		// List<Classifier> classifs = pre.predictRanking(data);
//		// classifs.forEach(classif ->
//		// System.out.println(classif.getClass().getSimpleName()));
//
//		// Read data set
//		// DataSource source = new
//		// DataSource("metaData_small_allPerformanceValues_onlyProbing.arff");
//		// Instances data = source.getDataSet();
//		// List<Integer> targetAttributes = new ArrayList<Integer>();
//		// for (int i = 45; i < 67; i++) {
//		// targetAttributes.add(i);
//		// }
//		//
//		// Evaluate ranker
//		// List<Double> evaluationResults =
//		// EvaluationHelper.evaluateRegressionRanker(new RandomForestRanker(), data,
//		// targetAttributes);
//		// evaluationResults.forEach(result -> System.out.println(result));
//
//		// evaluationResults = EvaluationHelper.evaluateRanker(new
//		// LinearRegressionRanker(), data, targetAttributes);
//		// evaluationResults.forEach(result -> System.out.println(result));
//		//
//		// evaluationResults = EvaluationHelper.evaluateRanker(new REPTreeRanker(),
//		// data, targetAttributes);
//		// evaluationResults.forEach(result -> System.out.println(result));
//		//
//		// evaluationResults = EvaluationHelper.evaluateRanker(new M5PRanker(), data,
//		// targetAttributes);
//		// evaluationResults.forEach(result -> System.out.println(result));
//
//		// List<Double> evaluationResults = EvaluationHelper.evaluateRanker(new
//		// PairwiseComparisonRanker(), data, targetAttributes);
//		// evaluationResults.forEach(result -> System.out.println(result));
//		//
//		// evaluationResults = EvaluationHelper.evaluateRanker(new
//		// InstanceBasedLabelRankingRanker(), data, targetAttributes);
//		// evaluationResults.forEach(result -> System.out.println(result));
//		//
//		// evaluationResults = EvaluationHelper.evaluateRanker(new
//		// InstanceBasedLabelRankingKemenyYoung(), data, targetAttributes);
//		// evaluationResults.forEach(result -> System.out.println(result));
//		//
//		// evaluationResults = EvaluationHelper.evaluateRanker(new
//		// InstanceBasedLabelRankingKemenyYoungSQRTN(), data, targetAttributes);
//		// evaluationResults.forEach(result -> System.out.println(result));
//
//		// ********************************* SQL STUFF
//		// **********************************************************
//		// Class.forName("oracle.jdbc.driver.OracleDriver");
//
//		// MySQLAdapter dataBaseAdapter = new MySQLAdapter("isys-db.cs.upb.de", "hgraf",
//		// "pw", "hgraf");
//		// HashMap<String, Object> map = new HashMap<String, Object> ();
//		// map.put("dataset_id", 2);
//		// map.put("dataset_name", "test2");
//		// dataBaseAdapter.insert("dataset_ids", map);
//		// dataBaseAdapter.close();
//
//		// get a job from the data base
//		// CustomMySQLAdapter dataBaseAdapter = new
//		// CustomMySQLAdapter("isys-db.cs.upb.de", "hgraf", "pw", "hgraf");
//		// dataBaseAdapter.close();
//		// int job;
//		//
//		// // download data set
//		// Instances data = OpenMLHelper.getInstancesById(job);
//		//
//		// // compute meta features
//		// GlobalCharacterizer chara = new GlobalCharacterizer();
//		// Map<String, Double> metaFeatures = chara.characterize(data);
//		//
//		// // insert into data base
//		// dataBaseAdapter = new CustomMySQLAdapter("isys-db.cs.upb.de", "hgraf", "pw",
//		// "hgraf");
//		// dataBaseAdapter.insert_noNewValues("metafeauture_values", metaFeatures);
//		// dataBaseAdapter.close();
//
//		// ******************************************* SQL STUFF
//		// ************************************************************
//		// CustomMySQLAdapter dataBaseAdapter= new
//		// CustomMySQLAdapter("isys-db.cs.upb.de", "hgraf", "wzq0A1GXAj9sxKVH",
//		// "hgraf");
//		// System.out.println("Open Adapter");
//		//
//		// // get arguments
//		// int offset = Integer.parseInt(args[0]);
//		// int index = Integer.parseInt(args[1]);
//		// int row = offset + index;
//		//
//		// // get data set id for job
//		// Map<String,String> conditions = new HashMap<String,String>();
//		// conditions.put("run_id", new Integer(row).toString());
//		// ResultSet results = dataBaseAdapter.getRowsOfTable("metafeature_runs",
//		// conditions);
//		// results.first();
//		// int dataset_id = results.getInt("dataset_id");
//		//
//		// // check status
//		// String status = results.getString("status");
//		// if (status.equals("created")) {
//		// // job is available
//		// Map<String, String> updateValues = new HashMap<String, String>();
//		// updateValues.put("status", "running");
//		// dataBaseAdapter.update("metafeature_runs", updateValues, conditions);
//		// dataBaseAdapter.close();
//		// System.out.println("Close Adapter");
//		//
//		// // get data set
//		// Instances instances = OpenMLHelper.getInstancesById(dataset_id);
//		// GlobalCharacterizer globalCharacterizer = new GlobalCharacterizer();
//		// List<Characterizer> characterizers = globalCharacterizer.getCharacterizers();
//		// for (Characterizer characterizer : characterizers) {
//		// // characterize
//		// StopWatch watch = new StopWatch();
//		// watch.start();
//		// Map<String, Double> values = characterizer.characterize(instances);
//		// watch.stop();
//		// Map<String, Object> times = new HashMap<String,Object>();
//		// times.put("run_id", row);
//		// times.put("metafeature_group",
//		// globalCharacterizer.getCharacterizerNamesMappings().get(characterizer));
//		// times.put("time", watch.getTime());
//		// try(CustomMySQLAdapter dataBaseAdapter3 = new
//		// CustomMySQLAdapter("isys-db.cs.upb.de", "hgraf", "wzq0A1GXAj9sxKVH",
//		// "hgraf");) {
//		// System.out.println("Open Adapter3");
//		// dataBaseAdapter3.insert_noNewValues("metafeature_times", times);
//		// // insert values
//		// values.forEach((metafeature, value)-> {
//		// Map<String, Object> insertValues = new HashMap<String, Object>();
//		// insertValues.put("run_id", row);
//		// insertValues.put("metafeature_name", metafeature);
//		// insertValues.put("metafeature_value", value);
//		// try {
//		// dataBaseAdapter3.insert_noNewValues("metafeature_values", insertValues);
//		// } catch (SQLException e) {
//		// // TODO Auto-generated catch block
//		// e.printStackTrace();
//		// }
//		//
//		// });
//		// dataBaseAdapter3.close();
//		// System.out.println("Close Adapter3");
//		// }
//		// }
//		//
//		// // put clean ending
//		// try ( CustomMySQLAdapter dataBaseAdapter4 = new
//		// CustomMySQLAdapter("isys-db.cs.upb.de", "hgraf", "wzq0A1GXAj9sxKVH",
//		// "hgraf");) {
//		// System.out.println("Open Adapter4");
//		// updateValues = new HashMap<String,String>();
//		// updateValues.put("status","finished");
//		// dataBaseAdapter4.update("metafeature_runs", updateValues, conditions);
//		// dataBaseAdapter4.close();
//		// System.out.println("Close Adapter4");
//		// }
//		// } else {
//		// // job is not available
//		// System.err.println("Couldn't get job");
//		// dataBaseAdapter.close();
//		// System.out.println("Close Adapter");
//		// }
//		//
//
//		// Instances data = OpenMLHelper.getInstancesById(dataId);
//		//
//		// GlobalCharacterizer globalChara = new GlobalCharacterizer();
//		// ArrayList<Characterizer> charas = globalChara.getCharacterizers();
//		// charas.forEach(characterizer -> {
//		//
//		// });
//
//		// CustomMySQLAdapter dataBaseAdapter = new
//		// CustomMySQLAdapter("isys-db.cs.upb.de", "hgraf", "wzq0A1GXAj9sxKVH",
//		// "hgraf");
//		//
//		// BufferedReader reader =
//		// Files.newBufferedReader(FileSystems.getDefault().getPath(Util.DATASET_INDEX_LARGE),
//		// Util.CHARSET);
//		// String line;
//		// while ((line = reader.readLine()) != null) {
//		// int did = Integer.parseInt(line);
//		// HashMap<String, Integer> map = new HashMap<String, Integer>();
//		// map.put("dataset_id", did);
//		// dataBaseAdapter.insert("metafeature_runs", map);
//		// }
//		//
//		// dataBaseAdapter.close();
//
//	}
//
//	public static void testjPL() throws TrainModelsFailedException, PredictionFailedException {
//		List<Integer> labels = Arrays.asList(0, 1);
//
//		List<Ranking> rankings = new ArrayList<>();
//		rankings.add(new Ranking(new int[] { 1, 0 }, Ranking.createCompareOperatorArrayForLabels(new int[] { 3 })));
//		rankings.add(new Ranking(new int[] { 1, 0 }, Ranking.createCompareOperatorArrayForLabels(new int[] { 3 })));
//		rankings.add(new Ranking(new int[] { 1, 0 }, Ranking.createCompareOperatorArrayForLabels(new int[] { 3 })));
//		rankings.add(new Ranking(new int[] { 1, 0 }, Ranking.createCompareOperatorArrayForLabels(new int[] { 3 })));
//		rankings.add(new Ranking(new int[] { 1, 0 }, Ranking.createCompareOperatorArrayForLabels(new int[] { 3 })));
//		rankings.add(new Ranking(new int[] { 1, 0 }, Ranking.createCompareOperatorArrayForLabels(new int[] { 3 })));
//		rankings.add(new Ranking(new int[] { 1, 0 }, Ranking.createCompareOperatorArrayForLabels(new int[] { 3 })));
//		rankings.add(new Ranking(new int[] { 0, 1 }, Ranking.createCompareOperatorArrayForLabels(new int[] { 3 })));
//
//		List<double[]> features = new ArrayList<>();
//		features.add(new double[] { -0.933919, 2.151642, -0.371715, -0.245385, 0.208949, 0.068299, 0.671798 });
//		features.add(new double[] { -0.933919, 2.151642, -0.371715, -0.245385, 0.208949, 0.068299, 0.671798 });
//		features.add(new double[] { -0.933919, 2.151642, -0.371715, -0.245385, 0.208949, 0.068299, 0.671798 });
//		features.add(new double[] { -0.933919, 2.151642, -0.371715, -0.245385, 0.208949, 0.068299, 0.671798 });
//		features.add(new double[] { -0.933919, 2.151642, -0.371715, -0.245385, 0.208949, 0.068299, 0.671798 });
//		features.add(new double[] { -0.933919, 2.151642, -0.371715, -0.245385, 0.208949, 0.068299, 0.671798 });
//		features.add(new double[] { -0.933919, 2.151642, -0.371715, -0.245385, 0.208949, 0.068299, 0.671798 });
//		features.add(new double[] { 0, 0, 0, 0, 0, 0, 0 });
//
//		LabelRankingDataset trainSet = new LabelRankingDataset(labels, features, rankings);
//
//		ArrayList<double[]> features_test = new ArrayList<double[]>();
//		features_test.add(new double[] { -0.933919, 2.151642, -0.371715, -0.245385, 0.208949, 0.068299, 0.671798 });
//		ArrayList<Ranking> rankings_test = new ArrayList<Ranking>();
//		// int[] trueLabels = { 1, 0 };
//		// int[] trueOperator = { Ranking.COMPARABLE_ENCODING };
//		// Ranking trueRanking = new Ranking(trueLabels, trueOperator);
//		// rankings_test.add(trueRanking);
//		rankings_test.add(null);
//
//		LabelRankingDataset testSet = new LabelRankingDataset(labels, features_test, rankings_test);
//
//		RandomGenerator.initializeRNG(1234);
//		LabelRankingByPairwiseComparisonLearningAlgorithm algo = new LabelRankingByPairwiseComparisonLearningAlgorithm();
//		LogisticRegressionWEKA baseLearnerAlgorithm = new LogisticRegressionWEKA();
//		algo.getAlgorithmConfiguration().setBaseLearnerAlgorithm(baseLearnerAlgorithm);
//		LabelRankingByPairwiseComparisonLearningModel learningModel = algo.train(trainSet);
//		System.out.println(learningModel.predict(testSet));
//	}
//
//	public static HashMap<String, List<Double>> readMetaFeatureGroupTimes() throws IOException {
//		BufferedReader readr = Files.newBufferedReader(
//				FileSystems.getDefault().getPath("data", "metafeature_computation_statistics","MetaStats.txt"),
//				Util.CHARSET);
//
//		String line;
//		HashMap<String, List<Double>> metaFeatureGroupTimes = new HashMap<String, List<Double>>();
//		String currentmetaFeatureGroupName = null;
//		List<Double> values = null;
//
//		while ((line = readr.readLine()) != null) {
//			if (line.equals("")) {
//				metaFeatureGroupTimes.put(currentmetaFeatureGroupName, values);
//				continue;
//			}
//
//			try {
//				values.add(Double.parseDouble(line.substring(0, line.length()-1)));
//			} catch (NumberFormatException e) {
//				currentmetaFeatureGroupName = line.replace(", 2 folds", "");
//				 values = new ArrayList<Double>();
//			}
//		}
//
//		return metaFeatureGroupTimes;
//	}
//
//}
