package experiments;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.sql.SQLException;

import dataHandling.mySQL.MetaDataDataBaseConnection;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;
import weka.core.converters.ArffSaver;

public class DataGetter {

	public static void dataget(String[] args) throws SQLException, IOException {
		MetaDataDataBaseConnection connection = new MetaDataDataBaseConnection(args[0], args[1], args[2], args[3]);

		System.out.println("getting classifier values");
		Instances classifiervalues = connection.getClassifierPerformancesForDataSetSet("all", "all-standard_config");
		classifiervalues.deleteAttributeAt(0);
		System.out.println("getting metadata");
		Instances metadata = connection.getMetaDataSetForDataSetSet("all", "all");

		System.out.println("merging");
		Instances complete = Instances.mergeInstances(metadata, classifiervalues);

		System.out.println("saving");
		ArffSaver saver = new ArffSaver();
		saver.setInstances(complete);
		saver.setFile(new File("complete.arff"));
		saver.writeBatch();
	}

	public static void dataSplit() throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader("complete_noid.arff"));
		ArffReader arff = new ArffReader(reader);
		Instances data = arff.getData();
		
		int deleteFromIndex = data.numAttributes() - 22;
		
		for (int i = 0; i < 22; i++) {
			Instances newInst = new Instances(data);
			
			int offset = 0;
			for (int del = 0; del < 22; del++) {
				if (del != i) {
					newInst.deleteAttributeAt(offset + deleteFromIndex);
				}else {
					offset++;
				}			
			}
			
			saveInstances(newInst, newInst.attribute(deleteFromIndex).name() + ".arff");
		}
	}
	
	public static void saveInstances(Instances instances, String name) throws IOException {
		ArffSaver saver = new ArffSaver();
		saver.setInstances(instances);
		saver.setFile(new File(name));
		saver.writeBatch();
	}
	
	public static void classifierPrint() throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader("complete_noid.arff"));
		ArffReader arff = new ArffReader(reader);
		Instances data = arff.getData();
		
		for (int i = data.numAttributes()-22; i < data.numAttributes(); i++) {
			System.out.print(data.attribute(i).name()+",");
		}
	}

	public static void main(String[] args) throws IOException {
		classifierPrint();
	}

}
