package ranker;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import weka.core.Instances;
import weka.core.converters.ArffSaver;

public class Main {

	public static void main(String[] args) throws Exception {

//		BufferedReader reader = Files.newBufferedReader(Util.resultsPath.resolve("Perf").resolve("metafeatures.arff"),
//				Util.charset);
//		Instances instances = new Instances(reader);
//		RandomForestRanker ranker = new RandomForestRanker();
//		Util.testRanker(ranker, instances);
		
		List<Integer> datasets = Util.getDataSetsFromIndex();
		Instances instances = Util.computeMetaFeatures(datasets);
		ArffSaver saver = new ArffSaver();
		saver.setInstances(instances);
		try {
			saver.setFile(new File(instances.relationName()));
			saver.writeBatch();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
