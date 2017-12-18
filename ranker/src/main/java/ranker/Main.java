package ranker;

import org.openml.apiconnector.settings.Settings;

public class Main {

	public static void main(String[] args) throws Exception {
		Settings.CACHE_DIRECTORY = Util.cacheDirectory.toString();
		Util.performanceMeasures(args);
	}

}
