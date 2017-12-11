package ranker;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

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

}
