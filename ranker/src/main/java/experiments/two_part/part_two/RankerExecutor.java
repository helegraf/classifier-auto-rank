package experiments.two_part.part_two;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;
import java.util.TreeMap;
import java.util.stream.Collectors;

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

import experiments.two_part.CSVHandler;
import ranker.core.algorithms.Ranker;
import ranker.core.algorithms.decomposition.DecompositionRanker;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public abstract class RankerExecutor {

	private Logger logger = LoggerFactory.getLogger(RankerExecutor.class);

	private static final String TRAIN_FILE_OPT = "trainfile";
	private static final String TEST_FILE_OPT = "testfile";
	private static final String TARGET_ATTS_OPT = "targets";
	private static final String RANKER_CONFIG_FILE_OPT = "rankconfig";
	private static final String OUTPUT_CONFIG_FILE_OPT = "outputconfig";
	

	protected void evaluateRankerWithArgs(String[] args) throws Exception {
		CommandLine cl = generateCommandLine(generateOptions(), args);
	
		Instances train = loadInstances(cl.getOptionValue(TRAIN_FILE_OPT));
		Instances test = loadInstances(cl.getOptionValue(TEST_FILE_OPT));
		List<Integer> targetAttributes = Arrays.stream(cl.getOptionValue(TARGET_ATTS_OPT)
				.substring(1, cl.getOptionValue(TARGET_ATTS_OPT).length() - 1).split(","))
				.mapToInt(Integer::parseInt).boxed().collect(Collectors.toList());
		RankerConfig rankerConfiguration = getConfigFromFile(getRankerConfigClass(), cl.getOptionValue(RANKER_CONFIG_FILE_OPT));
		CSVOutputConfig outputConfiguration = getConfigFromFile(CSVOutputConfig.class, cl.getOptionValue(OUTPUT_CONFIG_FILE_OPT));
	
		evaluateRanker(train, test, targetAttributes, rankerConfiguration, outputConfiguration);
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
		final Option targetAttributesOption = Option.builder("a").required().hasArg().longOpt(TARGET_ATTS_OPT)
				.desc("Target attributes in training and test data. Format [att1,att2,...,attn].").build();
		final Option configFileOption = Option.builder("rc").required().hasArg().longOpt(RANKER_CONFIG_FILE_OPT)
				.desc("Config file (for ranker) location").build();
		final Option outputConfigGileOption = Option.builder("oc").required().hasArg().longOpt(OUTPUT_CONFIG_FILE_OPT)
				.desc("Ouput file configuration").build();
	
		final Options options = new Options();
		options.addOption(trainFileOption);
		options.addOption(testFileOption);
		options.addOption(configFileOption);
		options.addOption(targetAttributesOption);
		options.addOption(outputConfigGileOption);
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
		pw.print("\n=====");
		pw.print("USAGE");
		pw.print("=====");
		formatter.printUsage(pw, 80, syntax, options);
		pw.flush();
	}

	protected void evaluateRanker(Instances train, Instances test, List<Integer> targetAttributes,
			RankerConfig configuration, CSVOutputConfig outputConfig) throws Exception {

		Ranker ranker = instantiate(configuration);
		
		long startTime = System.currentTimeMillis();
		ranker.buildRanker(train, targetAttributes);
		final long trainingTime = System.currentTimeMillis() - startTime;

		CSVHandler handler = new CSVHandler(outputConfig);

		test.forEach(instance -> {
			List<String> predictedRanking = null;
			List<Double> predictedValues = null;

			String instanceIdentifier = "";
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

			List<String> trueRanking = new ArrayList<>(targetAttributes.size());
			List<Double> trueValues = new ArrayList<>(targetAttributes.size());
			TreeMap<Double, String> trueValuePairs = new TreeMap<>();
			targetAttributes.forEach(attribute -> {
				double value = instance.value(attribute);
				String item = instance.attribute(attribute).name();
				trueValuePairs.put(value, item);
			});
			trueValuePairs.descendingMap().forEach((value, item) -> {
				trueRanking.add(item);
				trueValues.add(value);
			});

			handler.addRecord(instanceIdentifier, trueRanking, trueValues, predictedRanking, predictedValues,
					trainingTime, predictionTime);
		});

		handler.writeFile(outputConfig.getOutFilePath());
	}

	protected abstract Ranker instantiate(RankerConfig configuration);
}
