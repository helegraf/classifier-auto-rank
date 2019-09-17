package experiments.two_part;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.aeonbits.owner.ConfigCache;
import org.slf4j.LoggerFactory;

import jaicore.basic.SQLAdapter;
import jaicore.experiments.ExperimentDBEntry;
import jaicore.experiments.ExperimentRunner;
import jaicore.experiments.IExperimentDatabaseHandle;
import jaicore.experiments.IExperimentIntermediateResultProcessor;
import jaicore.experiments.IExperimentSetEvaluator;
import jaicore.experiments.databasehandle.ExperimenterSQLHandle;
import jaicore.experiments.exceptions.ExperimentDBInteractionFailedException;
import jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import jaicore.experiments.exceptions.IllegalExperimentSetupException;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

public class TwoPartExperimenter implements IExperimentSetEvaluator {

	private static final org.slf4j.Logger logger = LoggerFactory.getLogger(TwoPartExperimenter.class);
	private static TwoPartExperimenterConfig experimentConfig = ConfigCache
			.getOrCreate(TwoPartExperimenterConfig.class);

	public TwoPartExperimenter() {
		if (experimentConfig.evaluationsTable() == null) {
			throw new IllegalArgumentException("No evaluations table (db.evalTable) set in config");
		}
	}

	@Override
	public void evaluate(final ExperimentDBEntry experimentEntry,
			final IExperimentIntermediateResultProcessor processor) throws ExperimentEvaluationFailedException {
		try {

			Map<String, String> experimentValues = experimentEntry.getExperiment().getValuesOfKeyFields();

			logger.info("Evaluate Ranker {}", experimentValues.get("algorithm"));

			// Load the correct data
			BufferedReader reader = new BufferedReader(
					new FileReader("resources/" + experimentValues.get("metafeatures") + "_"
							+ experimentValues.get("imputation_variant") + ".arff"));
			ArffReader arff = new ArffReader(reader);
			Instances data = arff.getData();
			List<Integer> targetAttributes = detectTargetAttributes(data);

			int numFoldsValidation = Integer.parseInt(experimentValues.get("numFoldsValidation"));
			int numFoldValidation = Integer.parseInt(experimentValues.get("numFoldValidation"));
			int numFoldsTest = Integer.parseInt(experimentValues.get("numFoldsTest"));
			int numFoldTest = Integer.parseInt(experimentValues.get("numFoldTest"));

			Instances train = data.trainCV(numFoldsValidation, numFoldValidation);
			Instances validation = data.testCV(numFoldsValidation, numFoldValidation);
			Instances test = train.testCV(numFoldsTest, numFoldTest);
			train = train.trainCV(numFoldsTest, numFoldTest);

			// TODO execute jar

			// TODO read the evaluation results

			SQLAdapter adapter = new SQLAdapter(experimentConfig.getDBHost(), experimentConfig.getDBUsername(),
					experimentConfig.getDBPassword(), experimentConfig.getDBDatabaseName());

			Map<String, Object> results = new HashMap<>();
			// TODO add true rankings, predicted rankings, true values, and identifier for
			// instance for which each ranking was made
			processor.processResults(results);

			logger.info("Experiment done.");
		} catch (Exception e) {
			throw new ExperimentEvaluationFailedException(e);
		}
	}

	private List<Integer> detectTargetAttributes(Instances data) {
		ArrayList<Integer> result = new ArrayList<>();

		for (int i = 0; i < data.numAttributes(); i++) {
			if (data.attribute(i).name().startsWith("weka.")) {
				result.add(i);
			}
		}

		return result;
	}

	public static void main(String[] args)
			throws ExperimentDBInteractionFailedException, IllegalExperimentSetupException {
		IExperimentDatabaseHandle dbHandle = new ExperimenterSQLHandle(experimentConfig);
		ExperimentRunner runner = new ExperimentRunner(experimentConfig, new TwoPartExperimenter(), dbHandle);
		runner.randomlyConductExperiments(1, false);
	}
}
