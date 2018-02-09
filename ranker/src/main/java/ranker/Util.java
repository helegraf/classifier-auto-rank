package ranker;

import java.nio.charset.Charset;
import java.nio.file.FileSystems;
import java.nio.file.Path;

/**
 * Helper class with various methods to facilitate data generation and analysis.
 * 
 * @author Helena Graf
 *
 */
public class Util {
	public static Charset charset = Charset.forName("UTF-8");
	public static Path dataSetIndexPath = FileSystems.getDefault().getPath("datasets_100_1000");
	public static Path resultsPath = FileSystems.getDefault().getPath("data");
	public static Path cacheDirectory = FileSystems.getDefault().getPath("data");
	public static final String RANKER_BUILD_TIMES = "RankerBuildTimes";
	public static final String RANKER_PREDICT_TIMES = "RankerPredictTimes";
	public static final String DATA_ID = "DataId";
}
