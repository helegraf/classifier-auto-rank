package ranker.core.evaluation;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import ranker.Util;
import ranker.core.algorithms.Ranker;
import ranker.util.openMLUtil.OpenMLHelper;
import ranker.util.wekaUtil.EstimationProcedure;
import ranker.util.wekaUtil.EvaluationMeasure;
import ranker.util.wekaUtil.PredictiveAccuary;
import ranker.util.wekaUtil.StratifiedMCCV;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.functions.GaussianProcesses;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SGD;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.SimpleLinearRegression;
import weka.classifiers.functions.SimpleLogistic;
import weka.classifiers.functions.VotedPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.lazy.KStar;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.M5Rules;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.PART;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.LMT;
import weka.classifiers.trees.M5P;
import weka.classifiers.trees.REPTree;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

public class EvaluationHelper {
	
	/**
	 * Path for saving the generated jobs
	 */
	public static Path jobsPath = FileSystems.getDefault().getPath("jobs.txt");

	/**
	 * Evaluates all currently available measures for this type of ranker by means
	 * of leave one out on the given instances.
	 * 
	 * @param ranker
	 * @param instances
	 * @param targetAttributes
	 * @return
	 * @throws Exception
	 */
	public static List<Double> evaluateRanker(Ranker ranker, Instances instances, List<Integer> targetAttributes)
			throws Exception {
		// RegressionRanker ranker = new RegressionRanker();
		// ranker.buildRanker(instances);
		// ranker.predictRankingforInstance(instances.get(0)).forEach(action->System.out.println(action.getClass().getName()));
		// TODO cases RegressionRanker: include RMSE
		RankerEstimationProcedure estim = new LeaveOneOut();
		List<RankerEvaluationMeasure> measures = new ArrayList<RankerEvaluationMeasure>();
		measures.add(new KendallRankCorrelation());
		measures.add(new RootMeanSquaredError());
		Loss loss = new Loss();
		loss.setPerformanceOrder(PerformanceOrder.ASCENDING);
		measures.add(loss);
		BestThreeLoss bestLoss = new BestThreeLoss();
		bestLoss.setPerformanceOrder(PerformanceOrder.ASCENDING);
		measures.add(bestLoss);
		List<Double> result = estim.estimate(ranker, measures, instances, targetAttributes);
		System.out.println(estim.getDetailedEvaluationResults());
		return result;
	}

	/**
	 * Evaluates the predictive accuracy of the classifier on the data set by means
	 * of five times stratified MCCV with a 70%/30% train/test split. Return 0 if
	 * any errors ocurr during evaluation.
	 * 
	 * @param classifier
	 * @param dataset
	 * @return
	 */
	public static double evaluateClassifier(Classifier classifier, Instances dataset) {
		// Get result of evaluation
		double result;
		try {
			result = new StratifiedMCCV(5, 0.3).estimate(classifier, new PredictiveAccuary(), dataset);
		} catch (Exception e) {
			System.err.println(
					classifier.getClass().getSimpleName() + " could not be evaluated on " + dataset.relationName());
			System.err.println(e.getMessage());
			result = 0;
		}
		return result;
	}

	public static void generateJobs() throws IOException {
		BufferedReader reader = Files.newBufferedReader(Util.dataSetIndexPath, Util.charset);
		BufferedWriter writer = Files.newBufferedWriter(EvaluationHelper.jobsPath, Util.charset);
	
		String line = null;
		while ((line = reader.readLine()) != null) {
			for (Classifier classifier : EvaluationHelper.portfolio) {
				writer.write(classifier.getClass().getName() + "," + line);
				writer.newLine();
			}
		}
		reader.close();
		writer.close();
	}
	
	/**
	 * Executes the given jobs which must have been generated before.
	 * 
	 * @param args
	 * @throws IOException
	 * @throws Exception
	 */
	public static void executeJobs(String[] args) throws IOException, Exception {
		// Read input args
		if (args.length != 2) {
			throw new IllegalArgumentException("Wrong number of arguments supplied. Usage: Offset NumberOfJobs.");
		}
		int offset = Integer.parseInt(args[0]);
		int numberOfJobs = Integer.parseInt(args[1]);

		// Get jobs
		HashMap<Classifier, Integer> jobs = new HashMap<Classifier, Integer>();
		BufferedReader reader = Files.newBufferedReader(EvaluationHelper.jobsPath, Util.charset);
		String line = null;
		while (offset > 0 && (line = reader.readLine()) != null) {
			offset--;
		}
		while (numberOfJobs > 0 && (line = reader.readLine()) != null) {
			numberOfJobs--;
			String[] split = line.split(",");
			jobs.put(AbstractClassifier.forName(split[0], null), Integer.parseInt(split[1]));
		}
		reader.close();

		// Generate Performance Measures & save to files
		jobs.forEach((classifier, dataId) -> {
			try {
				Path dataPath = Util.resultsPath.resolve(classifier.getClass().getName() + "_" + dataId + ".txt");
				BufferedWriter writer = Files.newBufferedWriter(dataPath, Util.charset);
				Instances dataset = OpenMLHelper.getInstancesById(dataId);
				double result = evaluateClassifier(classifier, dataset);
				writer.write(Double.toString(result));
				writer.close();
			} catch (IOException e) {
				e.printStackTrace();
				System.err.println(e.getMessage());
			}
		});
	}

	/**
	 * Evaluates all classifiers on all data sets and saves results in a new data set (Features = Classifiers, Instances = data sets).
	 * 
	 * @param classifiers
	 * @param datasets
	 * @param evalM
	 * @param estimProc
	 * @param filepath
	 */
	public static void generatePerformanceMeasures(List<Classifier> classifiers, List<Instances> datasets,
			EvaluationMeasure evalM, EstimationProcedure estimProc, String filepath) {
		//TODO add data set id
	
		// Prepare table of results
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		for (int i = 0; i < datasets.size(); i++) {
			// Add index to attribute since they cannot have same names but some datasets do
			attributes.add(new Attribute(datasets.get(i).relationName() + " " + i));
		}
		Instances results = new Instances("PerformanceMeasures", attributes, 0);
	
		// Calculate performance of each classifier on each data set
		for (Classifier classifier : classifiers) {
			Instance instance = new DenseInstance(attributes.size());
			for (int i = 0; i < datasets.size(); i++) {
				try {
					double result = estimProc.estimate(classifier, evalM, datasets.get(i));
					instance.setValue(i, result);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
			results.add(instance);
		}
	
		ArffSaver saver = new ArffSaver();
		saver.setInstances(results);
		try {
			saver.setFile(new File(filepath));
			saver.writeBatch();
		} catch (IOException e) {
			e.printStackTrace();
		}
	
	}

	public static Classifier[] portfolio = { new BayesNet(), new NaiveBayes(), new NaiveBayesMultinomial(),
	new GaussianProcesses(), new LinearRegression(), new Logistic(), new MultilayerPerceptron(), new SGD(),
	new SMO(), new SimpleLinearRegression(), new SimpleLogistic(), new VotedPerceptron(), new IBk(),
	new KStar(), new DecisionTable(), new JRip(), new M5Rules(), new OneR(), new PART(), new ZeroR(),
	new DecisionStump(), new J48(), new LMT(), new M5P(), new RandomForest(), new RandomTree(), new REPTree() };

}
