package ranker.util.mySQL;

import java.io.BufferedReader;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.util.HashMap;
import java.util.Map;

import dataHandling.mySQL.CustomMySQLAdapter;
import ranker.Util;
import ranker.util.openMLUtil.OpenMLHelper;

public class mySQLHelper {
	
	/**
	 * Add data sets from the given index file to the list of data sets together with their name.
	 * 
	 * @param file The index file of the data sets
	 * @param host The host for the MySQL connection
	 * @param user The user name for the MySQL connection
	 * @param pw The password for the MySQL connection
	 * @param database The database for the MySQL connection
	 * @throws Exception If an Exception occurs while trying to connect to the database
	 */
	public void addDataSetsToIndex(String file, String host, String user, String pw, String database) throws Exception {
		CustomMySQLAdapter adapter = new CustomMySQLAdapter(host, user, pw, database);
		BufferedReader reader = Files.newBufferedReader(FileSystems.getDefault().getPath(file), Util.CHARSET);
		String line;
		while((line=reader.readLine())!=null) {
			Map<String, Object> map = new HashMap<String, Object>();
			int id = Integer.parseInt(line);
			String name = OpenMLHelper.getDataSetName(id);
			map.put("dataset_id", id);
			map.put("dataset_name", name);
			adapter.insert_noNewValues("dataset_ids", map);
		}
		
		adapter.close();
	}
}
