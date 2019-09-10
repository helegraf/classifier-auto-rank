package ranker.util.openMLUtil;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.apache.commons.io.IOUtils;
import org.openml.apiconnector.io.OpenmlConnector;
import org.openml.apiconnector.settings.Settings;
import org.openml.apiconnector.xml.Data;
import org.openml.apiconnector.xml.Data.DataSet;
import org.openml.apiconnector.xml.DataFeature;
import org.openml.apiconnector.xml.DataFeature.Feature;
import org.openml.apiconnector.xml.DataSetDescription;

import ranker.Util;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * @author Helena Graf
 *
 */
public class OpenMLHelper {

	public static String apiKey;

	/**
	 * Creates a list of data sets by id in a file with caps for the maximum of
	 * features and instances. Caps ignored if set to values smaller than 0.
	 * 
	 * @param maxNumFeatures  the maximal number of features (inclusive) the data
	 *                        set can have
	 * @param maxNumInstances the maximal number of instances (inclusive) the data
	 *                        set can have
	 * @param path            the path to which the index will be saved
	 * @throws Exception if the index cannot be retrieved from OpenML
	 */
	public static void createDataSetIndex(int maxNumFeatures, int maxNumInstances, Path path) throws Exception {
		// For statistics
		int unfiltered;
		int filteredBNG = 0;
		int filteredARFF = 0;
		int filteredTarget = 0;
		int filteredNumeric = 0;
		int fitForAnalysis = 0;

		// For saving data sets
		BufferedWriter writer = Files.newBufferedWriter(path, Util.CHARSET);

		// OpenML connection
		OpenmlConnector client = new OpenmlConnector();

		// Get data sets that are active
		HashMap<String, String> map = new HashMap<String, String>();
		map.put("status", "active");
		Data data = client.dataList(map);
		DataSet[] data_raw = data.getData();
		unfiltered = data_raw.length;

		// Filter out data sets not fit for analysis
		for (int i = 0; i < data_raw.length; i++) {
			// Keep track of progress to see if something freezes
			System.out.println("Progress: " + (Math.round(i * 1.0 / data_raw.length * 100.0)));

			// No generated streaming data
			if (data_raw[i].getName().contains("BNG")) {
				filteredBNG++;
				continue;
			}

			// No non-.ARFF files
			if (!data_raw[i].getFormat().equals("ARFF")) {
				filteredARFF++;
				continue;
			}

			// Analyze features
			DataFeature dataFeature = client.dataFeatures(data_raw[i].getDid());
			Feature[] features = dataFeature.getFeatures();
			if (maxNumFeatures > 0 && features.length > maxNumFeatures) {
				continue;
			}

			boolean noTarget = true;
			boolean numericTarget = true;
			for (int j = features.length - 1; j >= 0; j--) {
				if (features[j].getIs_target()) {
					noTarget = false;
					if (features[j].getDataType().equals("nominal")) {
						numericTarget = false;
					}
					break;
				}
			}

			// Analyze instances
			String numInst = data_raw[i].getQualityMap().get("NumberOfInstances");
			if (numInst == null) {
				System.out.println("Couldn't get num inst");
			} else {
				if (Double.parseDouble(numInst) > maxNumInstances) {
					continue;
				}
			}

			// No non-existent target attributes
			if (noTarget) {
				filteredTarget++;
				continue;
			}

			// No numeric target attributes
			if (numericTarget) {
				filteredNumeric++;
				continue;
			}

			// Data is fit for analysis, save
			writer.write(Integer.toString(data_raw[i].getDid()));
			writer.newLine();
			fitForAnalysis++;

		}

		writer.close();

		// Print statistics
		System.out.println("Unfiltered: " + unfiltered);
		System.out.println("BNG: " + filteredBNG);
		System.out.println("ARFF: " + filteredARFF);
		System.out.println("No target: " + filteredTarget);
		System.out.println("Numeric target: " + filteredNumeric);
		System.out.println("Fit for analysis: " + fitForAnalysis);
	}

	/**
	 * Get the ids of the data sets from a previously created index.
	 * 
	 * @param path the path of the index
	 * @return the ids
	 * @throws Exception if the index is not found
	 */
	public static List<Integer> getDataSetsFromIndex(Path path) throws Exception {
		List<Integer> dataSets = new ArrayList<Integer>();
		BufferedReader reader = Files.newBufferedReader(path, Util.CHARSET);
		String line = null;
		while ((line = reader.readLine()) != null) {
			int dataSetId = Integer.parseInt(line);
			dataSets.add(dataSetId);
		}
		return dataSets;
	}

	/**
	 * Downloads the data set with the given id and returns the Instances file for
	 * it. Will save the {@link org.openml.apiconnector.xml.DataSetDescription} and
	 * the Instances to the location specified in the
	 * {@link org.openml.apiconnector.settings.Settings} Class.
	 * 
	 * @param dataId the data set id
	 * @return the data set in WEKA Instances form
	 * @throws IOException if the data set cannot be retrieved from OpenML
	 */
	@SuppressWarnings("deprecation")
	public static Instances getInstancesById(int dataId) throws IOException {
		// Set the cache according to specified directory
		Settings.CACHE_DIRECTORY = Util.OPENML_CACHE_FOLDER;

		Instances dataset = null;

		// Get apiKey if not given
		if (OpenMLHelper.apiKey == null) {
			ClassLoader classLoader = Thread.currentThread().getContextClassLoader();
			InputStream inputStream = classLoader.getResourceAsStream(Util.APIKEY);
			// BufferedReader reader =
			// Files.newBufferedReader(FileSystems.getDefault().getPath(Util.APIKEY),
			// Util.CHARSET);
			OpenMLHelper.apiKey = IOUtils.toString(inputStream);
		}

		// Get dataset from OpenML
		OpenmlConnector client = new OpenmlConnector();
		try {
			DataSetDescription description = client.dataGet(dataId);
			File file = description.getDataset(apiKey);
			// Instances convert
			DataSource source = new DataSource(file.getCanonicalPath());
			dataset = source.getDataSet();
			dataset.setClassIndex(dataset.numAttributes() - 1);
			Attribute targetAttribute = dataset.attribute(description.getDefault_target_attribute());
			dataset.setClassIndex(targetAttribute.index());
		} catch (Exception e) {
			// These are IOExceptions anyways in the extended sense of this method
			throw new IOException(e.getMessage());
		}
		return dataset;
	}

	/**
	 * Gets the data set name for a given data set id on OpenML.
	 * 
	 * @param dataSetId The id of a data set on OpenML
	 * @return The corresponding name of the data set
	 * @throws Exception If something goes wrong while connecting to OpenML
	 */
	public static String getDataSetName(int dataSetId) throws Exception {
		Settings.CACHE_DIRECTORY = Util.OPENML_CACHE_FOLDER;

		OpenmlConnector client = new OpenmlConnector();
		DataSetDescription description = client.dataGet(dataSetId);
		return description.getName();
	}

}
