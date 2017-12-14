package ranker;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.nio.charset.Charset;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;

import org.openml.apiconnector.io.OpenmlConnector;
import org.openml.apiconnector.xml.Data;
import org.openml.apiconnector.xml.DataSetDescription;
import org.openml.apiconnector.xml.Data.DataSet;

import weka.classifiers.Classifier;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Main {

	public static void main(String[] args) throws Exception {
		
		Util.getDataFromOpenML();
		Util.generateJobs();
		
		Charset charset = Charset.forName("UTF-8");
		Path path2 = FileSystems.getDefault().getPath("src/main/ressources", "jobs.txt");
		BufferedReader reader2 = Files.newBufferedReader(path2,charset);
		String line = null;
		while((line = reader2.readLine()) != null) {
			System.out.println(line);
		}
		reader2.close();
		
		// TODO this should not be done here but in the datasets getter from OpenML!
		// data.setClassIndex(data.numAttributes() - 1);

		// Generate Performance Measures
		// Performance Measure Method has to be changed to accept string instead of
		// datasets
		// ranker.Util.generatePerformanceMeasures(classifiers, datasets,
		// predictiveAccuracy, estimproc);

		// TODO break big performance measures table down into many small tables with
		// attributes as metafeatures; last attribute (target) is performance

	}

}
