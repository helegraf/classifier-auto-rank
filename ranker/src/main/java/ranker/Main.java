package ranker;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

public class Main {

	public static void main(String[] args) throws Exception {

		// List<Integer> datasets = Util.getDataSetsFromIndex();
		// Instances instances1 = Util.getMetaFeaturesFromOpenML();
		// ArffSaver saver = new ArffSaver();
		// saver.setInstances(instances1);
		// try {
		// saver.setFile(new File(instances1.relationName() + ".arff"));
		// saver.writeBatch();
		// } catch (IOException e) {
		// e.printStackTrace();
		// }

		// BufferedReader reader =
		// Files.newBufferedReader(FileSystems.getDefault().getPath("meta_computed.arff"),
		// Util.charset);
		// Instances instances = new Instances(reader);
		// RegressionRanker ranker = new RandomForestRanker();
		// //System.out.println(Util.testRanker(ranker, instances));
		// ranker.buildRanker(instances);
		// ranker.predictRankingforInstance(instances.get(8));
		// System.out.println(ranker.getEstimatesForRanking());
		// ranker.predictRankingforInstance(instances.get(200));
		// System.out.println(ranker.getEstimatesForRanking());
		// ranker.predictRankingforInstance(instances.get(388));
		// System.out.println(ranker.getEstimatesForRanking());

		// ArrayList<Integer> dataIds = new ArrayList<Integer>();
		// BufferedReader reader =
		// Files.newBufferedReader(FileSystems.getDefault().getPath("datasets_100_1000"));
		// String line = null;
		// while ((line=reader.readLine())!=null) {
		// dataIds.add(Integer.parseInt(line));
		// }
		// Instances results = Util.computeMetaFeatures(dataIds);
		// ArffSaver saver = new ArffSaver();
		// saver.setInstances(results);
		// try {
		// saver.setFile(new File("meta_computed.arff"));
		// saver.writeBatch();
		// } catch (IOException e) {
		// e.printStackTrace();
		// }

		// Trying to save a .csv
		Path path = FileSystems.getDefault().getPath("datatest.csv");
		String seperator = ";";
		BufferedWriter writer = Files.newBufferedWriter(path, Util.charset);
		
		writer.write("1");
		writer.write(seperator);
		writer.write("2");
		writer.write(seperator);
		writer.write("3");
		writer.write(seperator);
		
		writer.newLine();
		
		writer.write("1");
		writer.write(seperator);
		writer.write("2");
		writer.write(seperator);
		writer.write("3");
		writer.write(seperator);
		
		writer.flush();
		writer.close();

	}
}
