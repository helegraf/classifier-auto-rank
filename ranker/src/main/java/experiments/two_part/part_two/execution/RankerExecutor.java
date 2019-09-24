package experiments.two_part.part_two.execution;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Properties;
import java.util.TreeMap;

import org.aeonbits.owner.Config;
import org.aeonbits.owner.ConfigFactory;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import experiments.two_part.part_two.output.OutputConfig;
import experiments.two_part.part_two.output.OutputHandler;
import ranker.core.algorithms.Ranker;
import ranker.core.algorithms.decomposition.DecompositionRanker;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public abstract class RankerExecutor {

	private Logger logger = LoggerFactory.getLogger(RankerExecutor.class);

	public static final String TRAIN_FILE_OPT = "trainfile";
	public static final String TEST_FILE_OPT = "testfile";
	public static final String RANKER_CONFIG_FILE_OPT = "rankconfig";
	public static final String OUTPUT_CONFIG_FILE_OPT = "outputconfig";
	public static final String SEED_OPT = "seed";
	public static final String OUT_FILE_NAME_OPT = "outfile";
	public static final String ACTIVE_CONFIG_FILE_OPT = "activeconfig";
	public static final String EXPERIMENT_ID_OPT = "experimentid";

	protected void evaluateRankerWithArgs(String[] args) throws Exception {
		CommandLine cl = generateCommandLine(generateOptions(), args);

		Instances train = loadInstances(cl.getOptionValue(TRAIN_FILE_OPT));
		Instances test = loadInstances(cl.getOptionValue(TEST_FILE_OPT));
		List<Integer> targetAttributes = new ArrayList<>();
		for (int i = 0; i < train.numAttributes(); i++) {
			if (train.attribute(i).name().startsWith("target:")) {
				targetAttributes.add(i);
			}
		}

		RankerConfig rankerConfiguration = getConfigFromFile(getRankerConfigClass(),
				cl.getOptionValue(RANKER_CONFIG_FILE_OPT));
		OutputConfig outputConfiguration = getConfigFromFile(OutputConfig.class,
				cl.getOptionValue(OUTPUT_CONFIG_FILE_OPT));
		String outFileName = cl.getOptionValue(OUT_FILE_NAME_OPT);
		String activeConfigFileName = cl.getOptionValue(ACTIVE_CONFIG_FILE_OPT);
		int experimentId = Integer.parseInt(cl.getOptionValue(EXPERIMENT_ID_OPT));

		// TODO parse seed also
		System.out.println("Running ranker with config");
		System.out.println(TRAIN_FILE_OPT + " " + cl.getOptionValue(TRAIN_FILE_OPT));
		System.out.println(TEST_FILE_OPT + " " + cl.getOptionValue(TEST_FILE_OPT));
		System.out.println(RANKER_CONFIG_FILE_OPT + " " + cl.getOptionValue(RANKER_CONFIG_FILE_OPT));
		System.out.println(OUTPUT_CONFIG_FILE_OPT + " " + cl.getOptionValue(OUTPUT_CONFIG_FILE_OPT));
		System.out.println(SEED_OPT + " " + cl.getOptionValue(SEED_OPT));
		System.out.println(OUT_FILE_NAME_OPT + " " + cl.getOptionValue(OUT_FILE_NAME_OPT));
		System.out.println(ACTIVE_CONFIG_FILE_OPT + " " + cl.getOptionValue(ACTIVE_CONFIG_FILE_OPT));
		System.out.println(EXPERIMENT_ID_OPT + " " + cl.getOptionValue(EXPERIMENT_ID_OPT));
		System.out.println("targets " + targetAttributes);

		evaluateRanker(train, test, targetAttributes, rankerConfiguration, outputConfiguration, outFileName,
				activeConfigFileName, experimentId);
	}

	protected Instances loadInstances(String path) throws Exception {
		DataSource source = new DataSource(path);
		return source.getDataSet();
	}

	protected <T extends Config> T getConfigFromFile(Class<? extends T> clazz, String configLocation) {
		Properties properties = new Properties();
		try {
			properties.load(new FileInputStream(new File(configLocation)));
			return ConfigFactory.create(clazz, properties);
		} catch (IOException e) {
			logger.warn(
					"Could not find configuration for ranker at location {}, trying to use standard configuration instead. Exception {}",
					configLocation, e);
		}
		return ConfigFactory.create(clazz);
	}

	protected abstract Class<? extends RankerConfig> getRankerConfigClass();

	private static Options generateOptions() {
		final Option trainFileOption = Option.builder("t").required().hasArg().longOpt(TRAIN_FILE_OPT)
				.desc("Training data location.").build();
		final Option testFileOption = Option.builder("T").required().hasArg().longOpt(TEST_FILE_OPT)
				.desc("Test data location.").build();
		final Option seedOption = Option.builder("s").required().hasArg().longOpt(SEED_OPT)
				.desc("Seed or split, depending on algorithm.").build();
		final Option configFileOption = Option.builder("rc").required().hasArg().longOpt(RANKER_CONFIG_FILE_OPT)
				.desc("Config file (for ranker) location").build();
		final Option outputConfigFileOption = Option.builder("oc").required().hasArg().longOpt(OUTPUT_CONFIG_FILE_OPT)
				.desc("Ouput file configuration").build();
		final Option outFileOption = Option.builder("of").required().hasArg().longOpt(OUT_FILE_NAME_OPT)
				.desc("Output file name (only name, not location)").build();
		final Option activeConfigFileOption = Option.builder("af").required().hasArg().longOpt(ACTIVE_CONFIG_FILE_OPT)
				.desc("A file to which the active configuration will be written").build();
		final Option experimentIdOption = Option.builder("id").required().hasArg().longOpt(EXPERIMENT_ID_OPT)
				.desc("Id of the current experiment, must be int.").build();

		final Options options = new Options();
		options.addOption(trainFileOption);
		options.addOption(testFileOption);
		options.addOption(configFileOption);
		options.addOption(seedOption);
		options.addOption(outputConfigFileOption);
		options.addOption(outFileOption);
		options.addOption(activeConfigFileOption);
		options.addOption(experimentIdOption);
		return options;
	}

	private static CommandLine generateCommandLine(final Options options, final String[] commandLineArguments)
			throws ParseException {
		final CommandLineParser cmdLineParser = new DefaultParser();

		try {
			return cmdLineParser.parse(options, commandLineArguments);
		} catch (ParseException e) {
			printUsage(options);
			throw e;
		}
	}

	private static void printUsage(final Options options) {
		final HelpFormatter formatter = new HelpFormatter();
		final String syntax = "Main";
		final PrintWriter pw = new PrintWriter(System.out);
		formatter.printUsage(pw, 80, syntax, options);
		pw.flush();
	}

	private Instances removeIdFromInstances(Instances data) {
		ArrayList<Attribute> attributes = new ArrayList<>(
				Collections.list(data.enumerateAttributes()).subList(1, data.numAttributes()));
		Instances newData = new Instances(data.relationName(), attributes, data.numInstances());
		data.forEach(instance -> {
			Instance newInstance = new DenseInstance(instance.numAttributes() - 1);
			for (int i = 1; i < instance.numAttributes(); i++) {
				newInstance.setValue(i - 1, instance.value(i));
			}
			newData.add(newInstance);
		});

		return newData;
	}

	private Instances nullAttributes(Instances instances, List<Integer> attributesToNull) {
		Instances newInst = new Instances(instances);
		newInst.clear();
		instances.forEach(instance -> {
			Instance newInstance = new DenseInstance(instance.weight(), Arrays.copyOf(instance.toDoubleArray(), instance.toDoubleArray().length));
			attributesToNull.forEach(attribute -> newInstance.setValue(attribute, 0.0));
			newInst.add(newInstance);			
		});
		
		return newInst;
	}

	protected void evaluateRanker(Instances train, Instances test, List<Integer> targetAttributes,
			RankerConfig configuration, OutputConfig outputConfig, String outFileName, String activeConfigFileName,
			int experimentId) throws Exception {

		Ranker ranker = instantiate(configuration);

		// TODO if ranker has hyperparameters to optimize, do a split, do hyperopt and
		// then eval - otherwise ONLY eval!

		// remove id from train and test
		List<Integer> reducedTargetAttributes = new ArrayList<>(targetAttributes.size());
		for (int i = 0; i < targetAttributes.size(); i++) {
			reducedTargetAttributes.add(targetAttributes.get(i) - 1);
		}
		Instances noIDTrain = removeIdFromInstances(train);
		Instances noIDTestTrueValues = removeIdFromInstances(test);
		Instances noIDTest = nullAttributes(noIDTestTrueValues, reducedTargetAttributes);

		long startTime = System.currentTimeMillis();
		ranker.buildRanker(noIDTrain, reducedTargetAttributes);
		final long trainingTime = System.currentTimeMillis() - startTime;

		// TODO after the ranker has been built for hyperopt the config needs to b added
		// check existence of directory before writing to it
		new File(outputConfig.getOutFilePath()).mkdirs();
		writeActiveConfig(this.getActiveConfiguration(), outputConfig.getOutFilePath() + "/" + activeConfigFileName);

		OutputHandler handler = new OutputHandler(outputConfig);

		for (int i = 0; i < noIDTest.numInstances(); i++) {
			Instance instance = noIDTest.get(i);

			List<String> predictedRanking = null;
			List<Double> predictedValues = null;

			String instanceIdentifier = String.valueOf(test.get(i).value(0));
			long predictionTime = System.currentTimeMillis();
			try {
				predictedRanking = ranker.predictRankingforInstance(instance);
				predictionTime = System.currentTimeMillis() - predictionTime;

				if (ranker instanceof DecompositionRanker) {
					predictedValues = ((DecompositionRanker) ranker).getEstimates();
				}
			} catch (Exception e) {
				logger.warn("No ranking for instance{}.", instanceIdentifier);
				predictionTime = System.currentTimeMillis() - predictionTime;
			}
			
			final int j = i;
			List<String> trueRanking = new ArrayList<>(reducedTargetAttributes.size());
			List<Double> trueValues = new ArrayList<>(reducedTargetAttributes.size());
			TreeMap<Double, List<String>> trueValuePairs = new TreeMap<>();
			reducedTargetAttributes.forEach(attribute -> {
				Instance trueinstance = noIDTestTrueValues.get(j);
				double value = trueinstance.value(attribute);
				String item = trueinstance.attribute(attribute).name();
				
				if (!trueValuePairs.containsKey(value)) {
					trueValuePairs.put(value, new ArrayList<String>());
				}
				
				trueValuePairs.get(value).add(item);
			});
			
			trueValuePairs.descendingMap().forEach((value, item) -> {
				trueRanking.addAll(item);
				item.forEach(itemm -> trueValues.add(value));
			});

			handler.addRecord(instanceIdentifier, trueRanking, trueValues, predictedRanking, predictedValues,
					trainingTime, predictionTime);
		}

		handler.writeFile(outputConfig.getOutFilePath(), outFileName);
	}

	private void writeActiveConfig(String activeConfiguration, String path) throws IOException {
		try (BufferedWriter writer = new BufferedWriter(new FileWriter(new File(path)))) {
			writer.write(activeConfiguration);
		}
	}

	protected abstract Ranker instantiate(RankerConfig configuration);

	protected abstract String getActiveConfiguration();
}
