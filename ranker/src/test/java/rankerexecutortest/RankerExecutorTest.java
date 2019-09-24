package rankerexecutortest;

import org.junit.Test;

import experiments.two_part.part_two.execution.AlorsRankerExecutor;
import experiments.two_part.part_two.execution.BestAlgorithmRankerExecutor;
import experiments.two_part.part_two.execution.MLPlanRegressionRankerExecutor;
import experiments.two_part.part_two.execution.RankClassificationRankerExecutor;
import experiments.two_part.part_two.execution.RankRegressionRankerExecutor;
import experiments.two_part.part_two.execution.RankerExecutor;
import experiments.two_part.part_two.execution.WEKARegressionRankerExecutor;

public class RankerExecutorTest {

	String trainFileLocation = "src/test/resources/executor_train.arff";
	String testFileLocation = "src/test/resources/executor_test.arff";
	String outputconfig = "conf/rankerconfigurations/outputconfig.properties";
	String seed = "0";
	String experimentId = "0";

	@Test
	public void testBestAlgorithmRanker() throws Exception {
		BestAlgorithmRankerExecutor.main(getCommandLine(trainFileLocation, testFileLocation, outputconfig,
				"conf/rankerconfigurations/BestAlgorithmRanker.properties", seed, "out_BestAlgorithmRanker",
				"activeConfig_BestAlgorithmRanker.properties", experimentId));
	}

	@Test
	public void testRankClassificationRanker() throws Exception {
		RankClassificationRankerExecutor.main(getCommandLine(trainFileLocation, testFileLocation, outputconfig,
				"conf/rankerconfigurations/RankClassificationRanker.properties", seed, "out_RankClassificationRanker",
				"activeConfig_RankClassificationRanker.properties", experimentId));
	}
	
	@Test
	public void testRankRegressionRanker() throws Exception {
		RankRegressionRankerExecutor.main(getCommandLine(trainFileLocation, testFileLocation, outputconfig,
				"conf/rankerconfigurations/RankRegressionRanker.properties", seed, "out_RankRegressionRanker",
				"activeConfig_RankRegressionRanker.properties", experimentId));
	}
	
	@Test
	public void testWEKARegressionRanker() throws Exception {
		WEKARegressionRankerExecutor.main(getCommandLine(trainFileLocation, testFileLocation, outputconfig,
				"conf/rankerconfigurations/WEKARegressionRanker.properties", seed, "out_WEKARegressionRanker",
				"activeConfig_WEKARegressionRanker.properties", experimentId));
	}
	
	@Test
	public void testAlorsRanker() throws Exception {
		AlorsRankerExecutor.main(getCommandLine(trainFileLocation, testFileLocation, outputconfig,
				"conf/rankerconfigurations/AlorsRanker.properties", seed, "out_AlorsRanker",
				"activeConfig_AlorsRanker.properties", experimentId));
	}
	
// deactivated because it takes too long	
//	@Test
//	public void testMLPlanRegressionRanker() throws Exception {
//		MLPlanRegressionRankerExecutor.main(getCommandLine(trainFileLocation, testFileLocation, outputconfig,
//				"conf/rankerconfigurations/AlorsRegressionRanker.properties", seed, "out_MLPlanRegressionRanker",
//				"activeConfig_MLPlanRegression.properties", experimentId));
//	}

	private String[] getCommandLine(String trainFileLocation, String testFileLocation, String outputconfig,
			String rankerconfig, String seed, String outfilename, String activeConfigFileName, String experimentId) {
		return new String[] { "-" + RankerExecutor.TRAIN_FILE_OPT, trainFileLocation,
				"-" + RankerExecutor.TEST_FILE_OPT, testFileLocation, "-" + RankerExecutor.OUTPUT_CONFIG_FILE_OPT,
				outputconfig, "-" + RankerExecutor.RANKER_CONFIG_FILE_OPT, rankerconfig, "-" + RankerExecutor.SEED_OPT,
				seed, "-" + RankerExecutor.OUT_FILE_NAME_OPT, outfilename, "-" + RankerExecutor.ACTIVE_CONFIG_FILE_OPT,
				activeConfigFileName, "-" + RankerExecutor.EXPERIMENT_ID_OPT, experimentId };
	}
}
