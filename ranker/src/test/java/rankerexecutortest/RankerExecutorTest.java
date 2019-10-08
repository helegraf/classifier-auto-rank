package rankerexecutortest;

import org.junit.Test;

import experiments.two_part.part_two.execution.AlorsRankerExecutor;
import experiments.two_part.part_two.execution.BestAlgorithmRankerExecutor;
import experiments.two_part.part_two.execution.RankClassificationRankerExecutor;
import experiments.two_part.part_two.execution.RankRegressionRankerExecutor;
import experiments.two_part.part_two.execution.RankerExecutor;
import experiments.two_part.part_two.execution.WEKARegressionRankerExecutor;

public class RankerExecutorTest {

	String trainFileLocation = "src/test/resources/executor_train.arff";
	String testFileLocation = "src/test/resources/executor_test.arff";
	String outputconfig = "conf/rankerconfigurations/outputconfig.properties";
	String experimentId = "0";
	String hyperSeed = "0";
	String hyperFoldNum = "1";
	String hyperNumFolds = "2";

	@Test
	public void testBestAlgorithmRanker() throws Exception {
		BestAlgorithmRankerExecutor.main(getCommandLine(trainFileLocation, testFileLocation, outputconfig,
				"conf/rankerconfigurations/BestAlgorithmRanker.properties", "out_BestAlgorithmRanker",
				"activeConfig_BestAlgorithmRanker.properties", experimentId, hyperSeed, hyperFoldNum, hyperNumFolds));
	}

	@Test
	public void testRankClassificationRanker() throws Exception {
		RankClassificationRankerExecutor.main(getCommandLine(trainFileLocation, testFileLocation, outputconfig,
				"conf/rankerconfigurations/RankClassificationRanker.properties", "out_RankClassificationRanker",
				"activeConfig_RankClassificationRanker.properties", experimentId, hyperSeed, hyperFoldNum,
				hyperNumFolds));
	}

	@Test
	public void testRankRegressionRanker() throws Exception {
		RankRegressionRankerExecutor.main(getCommandLine(trainFileLocation, testFileLocation, outputconfig,
				"conf/rankerconfigurations/RankRegressionRanker.properties", "out_RankRegressionRanker",
				"activeConfig_RankRegressionRanker.properties", experimentId, hyperSeed, hyperFoldNum, hyperNumFolds));
	}

	@Test
	public void testWEKARegressionRanker() throws Exception {
		WEKARegressionRankerExecutor.main(getCommandLine(trainFileLocation, testFileLocation, outputconfig,
				"conf/rankerconfigurations/WEKARegressionRanker.properties", "out_WEKARegressionRanker",
				"activeConfig_WEKARegressionRanker.properties", experimentId, hyperSeed, hyperFoldNum, hyperNumFolds));
	}

	@Test
	public void testAlorsRanker() throws Exception {
		AlorsRankerExecutor.main(getCommandLine(trainFileLocation, testFileLocation, outputconfig,
				"conf/rankerconfigurations/AlorsRanker.properties", "out_AlorsRanker",
				"activeConfig_AlorsRanker.properties", experimentId, hyperSeed, hyperFoldNum, hyperNumFolds));
	}

// deactivated because it takes too long	
//	@Test
//	public void testMLPlanRegressionRanker() throws Exception {
//		MLPlanRegressionRankerExecutor.main(getCommandLine(trainFileLocation, testFileLocation, outputconfig,
//				"conf/rankerconfigurations/AlorsRegressionRanker.properties", seed, "out_MLPlanRegressionRanker",
//				"activeConfig_MLPlanRegression.properties", experimentId));
//	}

	private String[] getCommandLine(String trainFileLocation, String testFileLocation, String outputconfig,
			String rankerconfig, String outfilename, String activeConfigFileName, String experimentId, String hyperSeed,
			String hyperFoldNum, String hyperNumFolds) {
		return new String[] { "-" + RankerExecutor.TRAIN_FILE_OPT, trainFileLocation,
				"-" + RankerExecutor.TEST_FILE_OPT, testFileLocation, "-" + RankerExecutor.OUTPUT_CONFIG_FILE_OPT,
				outputconfig, "-" + RankerExecutor.RANKER_CONFIG_FILE_OPT, rankerconfig,
				"-" + RankerExecutor.OUT_FILE_NAME_OPT, outfilename, "-" + RankerExecutor.ACTIVE_CONFIG_FILE_OPT,
				activeConfigFileName, "-" + RankerExecutor.EXPERIMENT_ID_OPT, experimentId,
				"-" + RankerExecutor.HYPEROPT_SEED_OPT, hyperSeed, "-" + RankerExecutor.HYPEROPT_FOLD_NUM_OPT,
				hyperFoldNum, "-" + RankerExecutor.HYPEROPT_NUM_FOLDS, hyperNumFolds };
	}
}
