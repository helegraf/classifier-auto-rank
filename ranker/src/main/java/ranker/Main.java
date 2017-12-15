package ranker;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;

import org.openml.apiconnector.settings.Settings;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instances;

public class Main {

	public static void main(String[] args) throws Exception {
		Settings.CACHE_DIRECTORY = FileSystems.getDefault().getPath("data").toAbsolutePath().toString();

		// Read input args
		if (args.length != 2) {
			throw new IllegalArgumentException("Wrong number of arguments supplied. Usage: Offset NumberOfJobs.");
		}
		int offset = Integer.parseInt(args[0]);
		int numberOfJobs = Integer.parseInt(args[1]);

		// Get jobs
		HashMap<Classifier, Integer> jobs = new HashMap<Classifier, Integer>();
		Charset charset = Charset.forName("UTF-8");
		Path path = FileSystems.getDefault().getPath("jobs.txt");
		BufferedReader reader = Files.newBufferedReader(path, charset);
		String line = null;
		while (offset > 0 && (line = reader.readLine()) != null) {
			offset--;
		}
		while (numberOfJobs > 0 && (line = reader.readLine()) != null) {
			numberOfJobs--;
			String[] split = line.split(",");
			jobs.put(AbstractClassifier.forName(split[0], null), Integer.parseInt(split[1]));
		}
		reader.close();

		// Generate Performance Measures & save to files
		jobs.forEach((classifier, dataId) -> {
			try {
				Path dataPath = FileSystems.getDefault().getPath("data",
						classifier.getClass().getSimpleName() + "_" + dataId + ".txt");

				BufferedWriter writer = Files.newBufferedWriter(dataPath, charset);
				Instances dataset = Util.getInstancesById(dataId);
				double result = Util.generatePerformanceMeasure(classifier, dataset);
				writer.write(Double.toString(result));
				writer.close();
			} catch (IOException e) {
				e.printStackTrace();
				System.err.println(e.getMessage());
			}
		});

	}

}
