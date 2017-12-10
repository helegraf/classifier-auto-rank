package ranker;

import java.util.List;
import java.util.Random;

import jaicore.ml.WekaUtil;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.Instance;

/**
 * Helper class with various methods to facilitate data generation and analysis.
 * 
 * @author Helena Graf
 *
 */
public class Util {

	public static void generatePerformanceMeasures(List<Classifier> classifiers, List<Instances> datasets,
			EvaluationMeasure evalM, EstimationProcedure estimProc, String filepath) {
		// TODO implements so that it accepts Strings instead of data sets

		// Prepare table of results
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		for (int i = 0; i < datasets.size(); i++) {
			// Add index to attribute since they cannot have same names but some datasets do
			// TODO better solution
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
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
			results.add(instance);
		}

		ArffSaver saver = new ArffSaver();
		saver.setInstances(results);
		try {
			saver.setFile(new File("./data/test.arff"));
			saver.writeBatch();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		// return "String";
	}

	/**
	 * Computes the performance of the classifier on the data set according to the
	 * evaluation measure by means of k times stratified Monte Carlo crossvalidation
	 * with a 70/30 split.
	 * 
	 * @param dataset
	 *            The dataset used for evaluation.
	 * @param classifier
	 *            The classifier used for evaluation.
	 * @param times
	 *            The number of times the classifier is evaluated.
	 * @param evaluationMeasure
	 *            The evaluation measure to use.
	 * @return The performance of the classifier.
	 * @throws Exception
	 *             If the evaluation of the classifier causes an Exception to be
	 *             thrown.
	 */
	public static double ktimesStratifiedMCCV(Instances dataset, Classifier classifier, int times,
			EvaluationMeasure evaluationMeasure) throws Exception {

		EstimationProcedure estimproc = (classif, evalM, data) -> {
			// TODO this should not be done here but in the datasets getter from OpenML!
			data.setClassIndex(data.numAttributes() - 1);
			double result = 0;
			for (int i = 0; i < times; i++) {
				List<Instances> splits = WekaUtil.getStratifiedSplit(data, new Random(i), 0.7);
				result += evalM.evaluate(classif, splits.get(0), splits.get(1));
			}
			
			result /= times;
			return result;
		};

		return estimproc.estimate(classifier, evaluationMeasure, dataset);
	}

	/**
	 * Computes the predicted accuracy of the classifier on the data set by means of
	 * k times stratified Monte Carlo crossvalidation with a 70/30 split.
	 * 
	 * @param dataset
	 *            The data set used for evaluation.
	 * @param classifier
	 *            The classifier used for evaluation.
	 * @param times
	 *            The number of times the classifier is evaluated.
	 * @return The predicted accuracy of the classifier.
	 * @throws Exception
	 *             If the evaluation of the classifier causes an Exception to be
	 *             thrown.
	 */
	public static double kTimesStratifiedMCCV(Instances dataset, Classifier classifier, int times) throws Exception {

		EvaluationMeasure predictiveAccuracy = (classif, train, test) -> {
			Evaluation evaluation = new Evaluation(train);
			classif.buildClassifier(train);
			evaluation.evaluateModel(classif, test);
			double result = evaluation.pctCorrect();
			return result;
		};

		return ktimesStratifiedMCCV(dataset, classifier, times, predictiveAccuracy);
	}

}
