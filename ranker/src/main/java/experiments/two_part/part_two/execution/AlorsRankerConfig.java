package experiments.two_part.part_two.execution;

public interface AlorsRankerConfig extends RankerConfig {

	@Key("cofi.executablePath")
	String executablePath();

	@Key("cofi.configurationPath")
	String configurationPath();

	@Key("cofi.outFolderPath")
	String outFolderPath();

	@Key("cofi.trainFilePath")
	String trainFilePath();

	@Key("cofi.testFilePath")
	String testFilePath();
}
