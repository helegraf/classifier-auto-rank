package datasets;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import org.junit.Test;

import weka.core.Instances;
import weka.core.converters.ArffSaver;

public class ASLibScenarioReaderTest {
	
	@Test
	public void testReadMinimalScenarioAndConvert() throws IOException {
		String location = "src/test/resources/ASLibParserTest/minimal";
		
		ASLibScenario scenario = new ASLibScenarioReader().readASLibScenario(location);
		
		Instances train = scenario.getTrainingDataForFold(1, 1, scenario.getPerformanceMeasures().get(1));
		Instances test = scenario.getTestingDataForFold(1, 1, scenario.getPerformanceMeasures().get(1));
		
		System.out.println("Training Data: ");
		System.out.println(train);
		
		System.out.println("Testing Data: ");
		System.out.println(test);
		
		ArffSaver saver = new ArffSaver();
		saver.setInstances(train);
		saver.setFile(new File(location + "/" + train.relationName() + ".arff"));
		saver.writeBatch();
		
		saver = new ArffSaver();
		saver.setInstances(test);
		saver.setFile(new File(location + "/" + test.relationName() + ".arff"));
		saver.writeBatch();
	}
	
	@Test
	public void testReadMinimalScenarioAndConvert2() throws IOException {
		String location = "src/test/resources/ASLibParserTest/minimal";
		
		ASLibScenario scenario = new ASLibScenarioReader().readASLibScenario(location);
		
		Instances train = scenario.getTrainingDataForFold(1, 2, scenario.getPerformanceMeasures().get(1));
		Instances test = scenario.getTestingDataForFold(1, 2, scenario.getPerformanceMeasures().get(1));
		
		System.out.println("Training Data: ");
		System.out.println(train);
		
		System.out.println("Testing Data: ");
		System.out.println(test);
		
		ArffSaver saver = new ArffSaver();
		saver.setInstances(train);
		saver.setFile(new File(location + "/" + train.relationName() + ".arff"));
		saver.writeBatch();
		
		saver = new ArffSaver();
		saver.setInstances(test);
		saver.setFile(new File(location + "/" + test.relationName() + ".arff"));
		saver.writeBatch();
	}
	
	@Test
	public void extensiveTest() throws IOException {
		File[] directories = new File("src/test/resources/ASLibParserTest/real/").listFiles(File::isDirectory);
		System.out.println("Testing set: " + Arrays.deepToString(directories));
		
		for (int i = 0; i < directories.length; i++) {
			System.err.println("Scenario " + directories[i]);
			ASLibScenario scenario = new ASLibScenarioReader().readASLibScenario(directories[i].getAbsolutePath());
			Instances train = scenario.getTrainingDataForFold(1);
			Instances test = scenario.getTestingDataForFold(1);
			
			ArffSaver saver = new ArffSaver();
			saver.setInstances(train);
			saver.setFile(new File(directories[i] + "/" + train.relationName() + ".arff"));
			saver.writeBatch();
			
			saver = new ArffSaver();
			saver.setInstances(test);
			saver.setFile(new File(directories[i] + "/" + test.relationName() + ".arff"));
			saver.writeBatch();
		}
	}
}
