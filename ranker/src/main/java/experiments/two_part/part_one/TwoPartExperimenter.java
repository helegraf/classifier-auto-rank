package experiments.two_part.part_one;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;
import java.util.stream.Collectors;

import org.aeonbits.owner.ConfigCache;
import org.aeonbits.owner.ConfigFactory;
import org.slf4j.LoggerFactory;

import experiments.two_part.part_two.execution.RankerExecutor;
import experiments.two_part.part_two.output.OutputConfig;
import experiments.two_part.part_two.output.OutputHandler;
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

public class TwoPartExperimenter implements IExperimentSetEvaluator {

	private static final org.slf4j.Logger logger = LoggerFactory.getLogger(TwoPartExperimenter.class);
	private static TwoPartExperimenterConfig experimentConfig = ConfigCache
			.getOrCreate(TwoPartExperimenterConfig.class);

	@Override
	public void evaluate(final ExperimentDBEntry experimentEntry,
			final IExperimentIntermediateResultProcessor processor) throws ExperimentEvaluationFailedException {
		try {
			Map<String, String> experimentValues = experimentEntry.getExperiment().getValuesOfKeyFields();
			String algorithm = experimentValues.get("algorithm");
			logger.info("Evaluate Ranker {}", algorithm);

			// get arguments (assume train and test datasets exist already)
			String fileLocation = experimentConfig.getDatasetFolder() + "/" +experimentValues.get("dataset") + "_repetition-"
					+ experimentValues.get("split_repetition") + "_fold-" + experimentValues.get("split_fold");
			String trainFileLocation = fileLocation + "_train.arff";
			String testFileLocation = fileLocation + "_test.arff";
			String outputconfig = "conf/rankerconfigurations/outputconfig.properties";
			String rankerconfig = "conf/rankerconfigurations/" + algorithm + ".properties";
			String rankerExecutable = "resources" + "/" + "rankerexecutables" + "/" + algorithm + ".jar";
			String baseFileName = algorithm + "_" + experimentValues.get("dataset") + "_repetition-"
					+ experimentValues.get("split_repetition") + "_fold-" + experimentValues.get("split_fold")
					+ System.currentTimeMillis();
			String outfileName = baseFileName + "_out.txt";
			String activeConfigFile = baseFileName + "_activeConfig.txt";

			// optional args
			String hyperoptSeed = experimentValues.get("hyperopt_seed");
			String hyperoptFoldNum = experimentValues.get("hyperopt_foldNum");
			String hyperoptNumFolds = experimentValues.get("hyperopt_numFolds");
			String hyperopt = "";
			if (hyperoptSeed != null && hyperoptFoldNum != null && hyperoptNumFolds != null) {
				hyperopt = String.format("-%s %s -%s %s -%s %s", RankerExecutor.HYPEROPT_SEED_OPT, hyperoptSeed,
						RankerExecutor.HYPEROPT_FOLD_NUM_OPT, hyperoptFoldNum, RankerExecutor.HYPEROPT_NUM_FOLDS,
						hyperoptNumFolds);
			}

			// if the ranker can upload intermediate results give it the experiment id
			if (algorithm.equals("MLPlanRegressionRanker") || algorithm.equals("AutoWEKARegressionRanker")) {
				Properties properties = new Properties();
				properties.load(new FileInputStream(new File(rankerconfig)));
				properties.setProperty("db.experiment_id", String.valueOf(experimentEntry.getId()));
				properties.store(new FileOutputStream(new File(rankerconfig)), "");
			}

			// execute jar
			Process process = new ProcessBuilder("java", "-jar", rankerExecutable, "-" + RankerExecutor.TRAIN_FILE_OPT,
					trainFileLocation, "-" + RankerExecutor.TEST_FILE_OPT, testFileLocation,
					"-" + RankerExecutor.OUTPUT_CONFIG_FILE_OPT, outputconfig,
					"-" + RankerExecutor.RANKER_CONFIG_FILE_OPT, rankerconfig,
					"-" + RankerExecutor.ACTIVE_CONFIG_FILE_OPT, activeConfigFile,
					"-" + RankerExecutor.EXPERIMENT_ID_OPT, String.valueOf(experimentEntry.getId()),
					"-" + RankerExecutor.OUT_FILE_NAME_OPT, outfileName, hyperopt).start();
			process.waitFor();
			InputStream errorstream  = process.getErrorStream();
			BufferedReader errorStreamReader = new BufferedReader(new InputStreamReader(errorstream));
			String line = errorStreamReader.readLine();
			while(line != null) {
				System.err.println(line);
				line = errorStreamReader.readLine();
			}

			// read the evaluation results
			String intermediateResultsTable = experimentConfig.getDBIntermediateTableName();
			try (SQLAdapter adapter = new SQLAdapter(experimentConfig.getDBHost(), experimentConfig.getDBUsername(),
					experimentConfig.getDBPassword(), experimentConfig.getDBDatabaseName())) {
				Properties properties = new Properties();
				properties.load(new FileInputStream(new File(outputconfig)));
				OutputConfig csvConfig = ConfigFactory.create(OutputConfig.class, properties);
				OutputHandler handler = new OutputHandler(csvConfig);
				handler.createIntermediateResultsTable(adapter, experimentConfig.getDBDatabaseName(),
						experimentConfig.getDBIntermediateTableName());
				handler.uploadFile(adapter, csvConfig.getOutFilePath(), outfileName, intermediateResultsTable,
						experimentEntry.getId());

				String activeConfiguration = "ERROR";
				try (BufferedReader reader = new BufferedReader(new FileReader(new File(csvConfig.getOutFilePath() + "/" + activeConfigFile)))) {
					activeConfiguration = reader.lines().collect(Collectors.joining());
				}

				Map<String, Object> results = new HashMap<>();
				results.put("done", true);
				results.put("active_configuration", activeConfiguration);
				processor.processResults(results);
			}

			logger.info("Experiment done.");
		} catch (Exception e) {
			throw new ExperimentEvaluationFailedException(e);
		}
	}

	public static void main(String[] args)
			throws ExperimentDBInteractionFailedException, IllegalExperimentSetupException {
		IExperimentDatabaseHandle dbHandle = new ExperimenterSQLHandle(experimentConfig);
		ExperimentRunner runner = new ExperimentRunner(experimentConfig, new TwoPartExperimenter(), dbHandle);
		runner.randomlyConductExperiments(1, false);
	}
}
