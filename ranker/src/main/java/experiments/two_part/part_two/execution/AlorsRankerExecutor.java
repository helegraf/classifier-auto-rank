package experiments.two_part.part_two.execution;

import alors.matrix_completion.cofirank.CofiConfig;
import ranker.core.algorithms.Ranker;
import ranker.core.algorithms.preference.ALORSRanker;

public class AlorsRankerExecutor extends RankerExecutor {

	@Override
	protected Class<? extends RankerConfig> getRankerConfigClass() {
		return AlorsRankerConfig.class;
	}

	@Override
	protected Ranker instantiate(RankerConfig configuration) {
		AlorsRankerConfig alorsconfig = (AlorsRankerConfig) configuration;
		CofiConfig coficonfig = new CofiConfig(alorsconfig.executablePath(), alorsconfig.configurationPath(),
				alorsconfig.outFolderPath(), alorsconfig.trainFilePath(), alorsconfig.testFilePath());
		return new ALORSRanker(coficonfig);
	}

	@Override
	protected String getActiveConfiguration() {
		// TODO Auto-generated method stub
		return "";
	}

	public static void main(String[] args) throws Exception {
		new AlorsRankerExecutor().evaluateRankerWithArgs(args);
	}

}
