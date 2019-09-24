package experiments.two_part.part_one;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;

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

			// get arguments
			String trainFileLocation = experimentValues.get("trainFile");
			String testFileLocation = experimentValues.get("testFile");
			String outputconfig = "conf/outputconfig.properties";
			String rankerconfig = "conf/rankerconfigurations/" + algorithm + ".properties";
			String rankerExecutable = "resources/rankerexecutables/" + algorithm + ".java";
			String intermediateResultsTable = experimentConfig.getDBIntermediateTableName();

			// if the ranker can upload intermediate results give it the experiment id
			if (algorithm.equals("MLPlanRegressionRanker") || algorithm.equals("AutoWEKARegressionRanker")) {
				Properties properties = new Properties();
				properties.load(new FileInputStream(new File(rankerconfig)));
				properties.setProperty("db.experiment_id", String.valueOf(experimentEntry.getId()));
				properties.store(new FileOutputStream(new File(rankerconfig)), "");
			}

			// execute jar
			//TODO add seed
			//TODO add outfile name for csv
			String outfilename = null;
			//TODo add outfile name for active configuration
			Process process = new ProcessBuilder("java -jar " + rankerExecutable, "-" + RankerExecutor.TRAIN_FILE_OPT,
					trainFileLocation, "-" + RankerExecutor.TEST_FILE_OPT, testFileLocation,
					"-" + RankerExecutor.OUTPUT_CONFIG_FILE_OPT, outputconfig,
					"-" + RankerExecutor.RANKER_CONFIG_FILE_OPT, rankerconfig).start();
			process.waitFor();

			// read the evaluation results
			try (SQLAdapter adapter = new SQLAdapter(experimentConfig.getDBHost(), experimentConfig.getDBUsername(),
					experimentConfig.getDBPassword(), experimentConfig.getDBDatabaseName())) {
				Properties properties = new Properties();
				properties.load(new FileInputStream(new File(outputconfig)));
				OutputConfig csvConfig = ConfigFactory.create(OutputConfig.class, properties);
				OutputHandler handler = new OutputHandler(csvConfig);
				handler.uploadFile(adapter, csvConfig.getOutFilePath(), outfilename,
						intermediateResultsTable, experimentEntry.getId());

				Map<String, Object> results = new HashMap<>();
				results.put("done", true);
				processor.processResults(results);
			}
			
			//TODO add active configuration to the table

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
