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

import ranker.Util;
import ranker.core.evaluation.EvaluationHelper;
import ranker.util.openMLUtil.OpenMLHelper;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class MetaFeatureHelper {

	/**
	 * Computes all global meta features for all data sets in the given list. Adds a
	 * feature for the data set id.
	 * 
	 * @param dataSetIds
	 * @return
	 * @throws Exception
	 */
	public static Instances computeMetaFeatures(List<Integer> dataSetIds) throws Exception {
	
		// Prepare List of Attributes for Instances
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		
		// Add the data set Id as an attribute
		attributes.add(new Attribute("OpenML Data Set ID"));
	
		// Qualities
		NoProbingCharacterizer characterizer = new NoProbingCharacterizer();
		String[] allDataQualities = characterizer.getIDs();
		Map<String, Integer> qualityIndices = new HashMap<String, Integer>();
		for (int i = 0; i < allDataQualities.length; i++) {
			String dataQuality = allDataQualities[i];
			qualityIndices.put(dataQuality, i+1);
			attributes.add(new Attribute(dataQuality));
		}
	
		HashMap<String, List<Double>> computationTimes = new HashMap<String, List<Double>>();
		for (String chara : characterizer.getCharacterizerNames()) {
			computationTimes.put(chara, new ArrayList<Double>());
		}
	
		// Classifiers
		Map<String, Integer> classifierIndices = new HashMap<String, Integer>();
		for (int i = 0; i < EvaluationHelper.portfolio.length; i++) {
			String classifierName = EvaluationHelper.portfolio[i].getClass().getName();
			classifierIndices.put(classifierName, allDataQualities.length + 1 + i);
			attributes.add(new Attribute(classifierName));
		}
	
		// Prepare instances
		Instances instances = new Instances("metaData_all_noProbing", attributes, 0);
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
		try (Stream<Path> paths = Files.walk(Util.resultsPath)) {
			paths.filter(Files::isRegularFile).forEach(file -> {
				try {
					BufferedReader reader = Files.newBufferedReader(file, Util.charset);
					String line = reader.readLine();
					if (line != null) {
						String fileName = file.getFileName().toString();
						fileName = fileName.substring(0, fileName.length() - 4);
						String[] parts = fileName.split("_");
						String classifierName = parts[0];
						int dataSetId = Integer.parseInt(parts[1]);
						if (dataSetIds.contains(dataSetId)) {
							if (classifierIndices.get(classifierName)!= null) {
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
		
		BufferedWriter writer = Files.newBufferedWriter(FileSystems.getDefault().getPath("MetaStats_new.txt"), Util.charset);
		for (String chara : computationTimes.keySet()) {
			List<Double> values = computationTimes.get(chara);
			writer.newLine();
			writer.write(chara);
			writer.newLine();
			values.forEach(value -> {
				try {
					writer.write(value + ";");
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

	public static Instances getMetaFeaturesFromOpenML() throws Exception {
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
		for (int i = 0; i < EvaluationHelper.portfolio.length; i++) {
			String classifierName = EvaluationHelper.portfolio[i].getClass().getName();
			classifierIndices.put(classifierName, allDataQualities.length + i);
			attributes.add(new Attribute(classifierName));
		}
	
		// Prepare instances
		Instances instances = new Instances("MetaDataOpenML", attributes, 0);
		HashMap<Integer, Instance> metaFeaturesForDataSets = new HashMap<Integer, Instance>();
	
		// Add meta features
		for (int dataSetId : OpenMLHelper.getDataSetsFromIndex()) {
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
		try (Stream<Path> paths = Files.walk(Util.resultsPath)) {
			paths.filter(Files::isRegularFile).forEach(file -> {
				try {
					BufferedReader reader = Files.newBufferedReader(file, Util.charset);
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
