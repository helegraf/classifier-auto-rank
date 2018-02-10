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

import org.apache.commons.math3.stat.inference.MannWhitneyUTest;
import org.apache.commons.math3.stat.ranking.NaNStrategy;
import org.apache.commons.math3.stat.ranking.TiesStrategy;

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
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SGD;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.SimpleLogistic;
import weka.classifiers.functions.VotedPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.lazy.KStar;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.PART;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.LMT;
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
		RankerEstimationProcedure estim = new LeaveOneOut();
		List<RankerEvaluationMeasure> measures = new ArrayList<RankerEvaluationMeasure>();
		measures.add(new KendallRankCorrelation());
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

	public static List<Double> evaluateRegressionRanker(Ranker ranker, Instances instances,
			List<Integer> targetAttributes) throws Exception {
		RankerEstimationProcedure estim = new LeaveOneOut();
		List<RankerEvaluationMeasure> measures = new ArrayList<RankerEvaluationMeasure>();
		measures.add(new KendallRankCorrelation());
		measures.add(new RootMeanSquareError());
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
	 * Evaluates all classifiers on all data sets and saves results in a new data
	 * set (Features = Classifiers, Instances = data sets).
	 * 
	 * @param classifiers
	 * @param datasets
	 * @param evalM
	 * @param estimProc
	 * @param filepath
	 */
	public static void generatePerformanceMeasures(List<Classifier> classifiers, List<Instances> datasets,
			EvaluationMeasure evalM, EstimationProcedure estimProc, String filepath) {
		// TODO add data set id

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

	/**
	 * For comparing whether one Ranker is significantly better regarding a measure
	 * than the other; to be used with the .csv files when a ranker is generated.
	 * 
	 * @param firstFile
	 *            Evaluation results of the first ranker
	 * @param secondFile
	 *            Evaluation results of the second ranker
	 * @param measure
	 *            The measure to compare
	 * @return
	 * @throws IOException
	 */
	public static double computeWhitneyU(Path firstFile, Path secondFile, String measure) throws IOException {
		// Get the values
		double[] xArray = getValues(firstFile, measure);
		double[] yArray = getValues(secondFile, measure);

		// Compute result
		MannWhitneyUTest uTest = new MannWhitneyUTest(NaNStrategy.REMOVED, TiesStrategy.AVERAGE);
		System.out.print("& " + uTest.mannWhitneyU(xArray, yArray) + " ");
		return uTest.mannWhitneyUTest(xArray, yArray);
	}

	public static double[] getValues(Path path, String measure) throws IOException {
		List<Double> values = new ArrayList<Double>();
		BufferedReader reader = Files.newBufferedReader(path, Util.charset);

		// Find index of measure
		String line = reader.readLine();
		int measureIndex = findIndex(line, measure);

		// Get contents
		while ((line = reader.readLine()) != null) {
			String[] contents = line.split(";");
			double value = Double.parseDouble(contents[measureIndex]);
			values.add(value);
		}

		return convertToArray(values);

	}

	public static int findIndex(String line, String measure) {
		int measureIndex = -1;
		String[] measures = line.split(";");
		for (int i = 0; i < measures.length; i++) {
			if (measures[i].replaceAll(" ", "").equals(measure)) {
				measureIndex = i;
				break;
			}
		}
		return measureIndex;
	}

	public static double[] convertToArray(List<Double> values) {
		double[] array = new double[values.size()];
		for (int i = 0; i < values.size(); i++) {
			array[i] = values.get(i);
		}
		return array;
	}

	public static Classifier[] portfolio = { new BayesNet(), new NaiveBayes(), new NaiveBayesMultinomial(),
			new Logistic(), new MultilayerPerceptron(), new SGD(), new SMO(), new SimpleLogistic(),
			new VotedPerceptron(), new IBk(), new KStar(), new DecisionTable(), new JRip(), new OneR(), new PART(),
			new ZeroR(), new DecisionStump(), new J48(), new LMT(), new RandomForest(), new RandomTree(),
			new REPTree() };

}
