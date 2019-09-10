package ranker.core.metafeatures;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import org.openml.apiconnector.io.OpenmlConnector;
import org.openml.apiconnector.settings.Settings;

import ranker.Util;
import ranker.util.openMLUtil.OpenMLHelper;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class MetaFeatureHelper {

	/**
	 * Compute a set of meta features for a number of data sets identified by their
	 * OpenML data set ID. Adds the OpenML ID of the data set as a feature.
	 * 
	 * @param dataSetIds      The IDs of the data sets
	 * @param characterizer   The characterizer used to compute the meta features
	 * @param metaDataSetName The name which will be given to the data set
	 * @return The data set containing the meta features
	 * @throws Exception if the meta features cannot be computed
	 */
	public static Instances computeMetaFeatures(List<Integer> dataSetIds, GlobalCharacterizer characterizer,
			String metaDataSetName) throws Exception {

		// Prepare List of Attributes for Instances
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();

		// Add the data set Id as an attribute
		attributes.add(new Attribute(Util.OPENML_DATASET_ID_FEATURE));

		// Qualities
		String[] allDataQualities = characterizer.getIDs();
		Map<String, Integer> qualityIndices = new HashMap<String, Integer>();
		for (int i = 0; i < allDataQualities.length; i++) {
			String dataQuality = allDataQualities[i];
			qualityIndices.put(dataQuality, i + 1);
			attributes.add(new Attribute(dataQuality));
		}

		HashMap<String, List<Double>> computationTimes = new HashMap<String, List<Double>>();
		for (String chara : characterizer.getCharacterizerNames()) {
			computationTimes.put(chara, new ArrayList<Double>());
		}

		// Classifiers
		Map<String, Integer> classifierIndices = new HashMap<String, Integer>();
		for (int i = 0; i < Util.PORTFOLIO.length; i++) {
			String classifierName = Util.PORTFOLIO[i].getClass().getName();
			classifierIndices.put(classifierName, allDataQualities.length + 1 + i);
			attributes.add(new Attribute(classifierName));
		}

		// Prepare instances
		Instances instances = new Instances(metaDataSetName, attributes, 0);
		HashMap<Integer, Instance> metaFeaturesForDataSets = new HashMap<Integer, Instance>();

		// Add meta features
		for (int dataSetId : dataSetIds) {
			Instance instance = new DenseInstance(attributes.size());
			// Add data set id
			instance.setValue(0, dataSetId);

			Instances openMLData = null;
			try {
				openMLData = OpenMLHelper.getInstancesById(dataSetId);
			} catch (IOException e) {
				System.out.println("Couldn't get " + dataSetId);
				continue;
			}

			Map<String, Double> dataQualities = characterizer.characterize(openMLData);

			Map<String, Double> times = characterizer.getMetaFeatureComputationTimes();
			System.out.println(times);
			for (String metaFeature : times.keySet()) {
				computationTimes.get(metaFeature).add(times.get(metaFeature));
			}

			dataQualities.forEach((dataQuality, value) -> {
				if (value != null) {
					instance.setValue(qualityIndices.get(dataQuality), value);
				}
			});
			metaFeaturesForDataSets.put(dataSetId, instance);
		}

		// Add performance results
		try (Stream<Path> paths = Files
				.walk(FileSystems.getDefault().getPath(Util.CLASSIFIER_EVALUATION_RESULTS_FOLDER))) {
			paths.filter(Files::isRegularFile).forEach(file -> {
				try {
					BufferedReader reader = Files.newBufferedReader(file, Util.CHARSET);
					String line = reader.readLine();
					if (line != null) {
						String fileName = file.getFileName().toString();
						fileName = fileName.substring(0, fileName.length() - 4);
						String[] parts = fileName.split(Util.CLASSIFIER_EVALUATION_RESULTS_SEPARATOR);
						String classifierName = parts[0];
						int dataSetId = Integer.parseInt(parts[1]);
						if (dataSetIds.contains(dataSetId)) {
							if (classifierIndices.get(classifierName) != null) {
								Instance instance = metaFeaturesForDataSets.get(dataSetId);
								int attIndex = classifierIndices.get(classifierName);
								double value = Double.parseDouble(line);
								instance.setValue(attIndex, value);
							}
						}
					}

				} catch (IOException e) {
					e.printStackTrace();
				}
			});
		}

		BufferedWriter writer = Files
				.newBufferedWriter(FileSystems.getDefault().getPath(Util.METAFEATURE_COMPUTATION_STATISTIC_FOLDER)
						.resolve(metaDataSetName + "_statistics_" + System.currentTimeMillis()), Util.CHARSET);
		for (String chara : computationTimes.keySet()) {
			List<Double> values = computationTimes.get(chara);
			writer.newLine();
			writer.write(chara);
			writer.newLine();
			values.forEach(value -> {
				try {
					writer.write(value + Util.CSV_SEPARATOR);
					writer.newLine();
				} catch (IOException e) {
					e.printStackTrace();
				}
			});
		}
		writer.flush();
		writer.close();

		instances.addAll(metaFeaturesForDataSets.values());
		return instances;
	}

	public static Instances gatherClassifierPerformanceValues(List<Integer> dataSetIds) throws IOException {
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("DatasetId"));

		// Classifiers mapping name to index in created instances
		Map<String, Integer> classifierIndices = new HashMap<String, Integer>();
		for (int i = 0; i < Util.PORTFOLIO.length; i++) {
			String classifierName = Util.PORTFOLIO[i].getClass().getName();
			classifierIndices.put(classifierName, i + 1);
			attributes.add(new Attribute(classifierName));
		}

		HashMap<Integer, Instance> metaFeaturesForDataSets = new HashMap<Integer, Instance>();
		for (int i : dataSetIds) {
			metaFeaturesForDataSets.put(i, new DenseInstance(classifierIndices.size() + 1));
			metaFeaturesForDataSets.get(i).setValue(0, i);
		}

		// Add performance results
		try (Stream<Path> paths = Files
				.walk(FileSystems.getDefault().getPath(Util.CLASSIFIER_EVALUATION_RESULTS_FOLDER))) {
			paths.filter(Files::isRegularFile).forEach(file -> {
				try {
					BufferedReader reader = Files.newBufferedReader(file, Util.CHARSET);
					String line = reader.readLine();
					if (line != null) {
						String fileName = file.getFileName().toString();
						fileName = fileName.substring(0, fileName.length() - 4);
						String[] parts = fileName.split(Util.CLASSIFIER_EVALUATION_RESULTS_SEPARATOR);
						String classifierName = parts[0];
						int dataSetId = Integer.parseInt(parts[1]);
						if (dataSetIds.contains(dataSetId)) {
							if (classifierIndices.get(classifierName) != null) {
								Instance instance = metaFeaturesForDataSets.get(dataSetId);
								int attIndex = classifierIndices.get(classifierName);
								double value = Double.parseDouble(line);
								instance.setValue(attIndex, value);
							}
						}
					}

				} catch (IOException e) {
					e.printStackTrace();
				}
			});
		}

		Instances instances = new Instances("classifiers", attributes, metaFeaturesForDataSets.size());
		instances.addAll(metaFeaturesForDataSets.values());
		return instances;
	}

	/**
	 * Gets the meta features from OpenML for all the data sets contained in the
	 * file located at the given Path.
	 * 
	 * @param dataSetIndices the indices for which to get meta features
	 * @return a data set containing the meta features for the data sets
	 * @throws Exception if the meta data cannot be retrieved
	 */
	public static Instances getMetaFeaturesFromOpenML(Path dataSetIndices) throws Exception {
		// Set the OpenML cache to the specified directory
		Settings.CACHE_DIRECTORY = Util.OPENML_CACHE_FOLDER;

		// Prepare List of Attributes for Instances
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();

		// Qualities
		OpenmlConnector client = new OpenmlConnector();
		String[] allDataQualities = client.dataQualitiesList().getQualities();
		Map<String, Integer> qualityIndices = new HashMap<String, Integer>();
		for (int i = 0; i < allDataQualities.length; i++) {
			String dataQuality = allDataQualities[i];
			qualityIndices.put(dataQuality, i);
			attributes.add(new Attribute(dataQuality));
		}

		// Classifiers
		Map<String, Integer> classifierIndices = new HashMap<String, Integer>();
		for (int i = 0; i < Util.PORTFOLIO.length; i++) {
			String classifierName = Util.PORTFOLIO[i].getClass().getName();
			classifierIndices.put(classifierName, allDataQualities.length + i);
			attributes.add(new Attribute(classifierName));
		}

		// Prepare instances
		Instances instances = new Instances("MetaDataOpenML", attributes, 0);
		HashMap<Integer, Instance> metaFeaturesForDataSets = new HashMap<Integer, Instance>();

		// Add meta features
		for (int dataSetId : OpenMLHelper.getDataSetsFromIndex(dataSetIndices)) {
			Instance instance = new DenseInstance(attributes.size());
			Map<String, String> dataQualities = client.dataQualities(dataSetId).getQualitiesMap();
			dataQualities.forEach((dataQuality, value) -> {
				if (value != null) {
					instance.setValue(qualityIndices.get(dataQuality), Double.parseDouble(value));
				}
			});
			metaFeaturesForDataSets.put(dataSetId, instance);
		}

		// Add performance results
		try (Stream<Path> paths = Files
				.walk(FileSystems.getDefault().getPath(Util.CLASSIFIER_EVALUATION_RESULTS_FOLDER))) {
			paths.filter(Files::isRegularFile).forEach(file -> {
				try {
					BufferedReader reader = Files.newBufferedReader(file, Util.CHARSET);
					String line = reader.readLine();
					String fileName = file.getFileName().toString();
					fileName = fileName.substring(0, fileName.length() - 4);
					String[] parts = fileName.split("_");
					String classifierName = parts[0];
					int dataSetId = Integer.parseInt(parts[1]);
					Instance instance = metaFeaturesForDataSets.get(dataSetId);
					int attIndex = classifierIndices.get(classifierName);
					double value;
					if (line != null) {
						value = Double.parseDouble(line);
					} else {
						value = 0;
					}
					instance.setValue(attIndex, value);

				} catch (IOException e) {
					e.printStackTrace();
				}
			});
		}

		instances.addAll(metaFeaturesForDataSets.values());
		return instances;
	}

}
