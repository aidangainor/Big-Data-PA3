package cosi129.pa3;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

import cosi129.pa3.StringIntegerList.StringInteger;

public class LemsToDoc {
	private final static String PROFESSIONS_FILE = "professions.txt";


	public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
		// TODO Auto-generated method stub
		Configuration conf = new Configuration();
		String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
		
		Job job = Job.getInstance(conf);
		job.setJarByClass(LemsToDoc.class);
		job.setMapperClass(ProfessionMapper.class); 
		job.setReducerClass(ProfessionReducer.class);

		job.setInputFormatClass(KeyValueTextInputFormat.class);
		conf.set("mapreduce.input.keyvaluelinerecordreader.key.value.separator","\t");
		job.setMapOutputValueClass(StringIntegerList.class);
		job.setMapOutputKeyClass(Text.class);
		//job.setOutputKeyClass(Text.class);
		//job.setOutputValueClass(Text.class);
		FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
		FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));
		System.exit(job.waitForCompletion(true)? 0: 1);
	}
	
	public static class ProfessionMapper extends Mapper<Text, Text, Text, StringIntegerList> {
        private HashMap<String, ArrayList<String>> professionMapping;
        
        // Setup builds hash map of names-to-professions before mapper jobs
        protected void setup(Context context) throws IOException, InterruptedException {
        	professionMapping = getProfessionMapping();
    	}
        
		public void map(Text articleId, Text indices, Context context) throws IOException, InterruptedException {
			// Check if this person has profession classified, if not ignore
			String articleIdTrimmed = articleId.toString().trim();
			if (professionMapping.containsKey(articleIdTrimmed)) {
				StringIntegerList wordFreqList = new StringIntegerList();
				wordFreqList.readFromString(indices.toString());
				ArrayList<String> professions = professionMapping.get(articleIdTrimmed);
				// For each profession this person belongs to, emit all (word, freq) pairs to it
				for (String profession : professions) {
					context.write(new Text(profession), wordFreqList);
				}
			}
		}
			
	}
	
	public static class ProfessionReducer extends Reducer<Text, StringIntegerList, Text, Text> {
		public void reduce(Text prof, Iterable<StringIntegerList> lists, Context context ) throws IOException{
			
		   Configuration config = new Configuration();
		   config.addResource(new Path("/etc/hadoop/conf/core-site.xml"));
		   config.addResource(new Path("/etc/hadoop/conf/hdfs-site.xml"));
		   config.set("fs.hdfs.impl", 
			            org.apache.hadoop.hdfs.DistributedFileSystem.class.getName()
			        );
			       config.set("fs.file.impl",
			            org.apache.hadoop.fs.LocalFileSystem.class.getName()
			        );
		  FileSystem dfs = FileSystem.get(config);
		  System.out.println(dfs.getWorkingDirectory() +" this is from /n/n");
		  String dir = dfs.getWorkingDirectory()+ "/prof-docs/"+ prof;
		  Path src = new Path(dir);
		  dfs.mkdirs(src);
		  
		  StringBuilder buffer; 
		  int id = 0;
		  BufferedWriter br;
		  Iterator iter = lists.iterator();
		  while(iter.hasNext()){
			  StringIntegerList sil = (StringIntegerList) iter.next();
			  buffer = new StringBuilder();
			  Path filePath = new Path(dir+"/"+id);
			  
			  br = new BufferedWriter(new OutputStreamWriter(dfs.create(filePath, true)));
			  for(StringInteger si : sil.getIndices()){
				 
				  String lemma = si.getString();
				  int freq = si.getValue();
				  for(int i = 0; i < freq; i++){
					  buffer.append(" " + lemma);
				  }
				  br.write(buffer.toString());
				  buffer.setLength(0);
			  }
			  br.close();
			  id++;
		  }
			
		}
	}
	
	public static HashMap<String, ArrayList<String>> getProfessionMapping() throws IOException {
		HashMap<String, ArrayList<String>> professionMapping = new HashMap<String, ArrayList<String>>();
		BufferedReader bufferedReader = null;
		try {
			InputStream profFile = LemmaIndexFormater.class.getResourceAsStream(PROFESSIONS_FILE);
			InputStreamReader profInStream = new InputStreamReader(profFile);
			bufferedReader = new BufferedReader(profInStream);
			String line;
			while ((line = bufferedReader.readLine()) != null) {
                String[] split = line.split(":");
                String name = "";
                // Separate names and their professions
                if(split.length > 0) name = split[0].trim();
                String[] professions = new String[1];
                if(split.length > 1 ) professions = split[1].split(","); 
                ArrayList<String> profList = new ArrayList<String>();
               
                for (String prof : professions) {
                    if(prof != null) profList.add(prof.trim());
                }
                professionMapping.put(name, profList);
			}
			
		} finally {
			try {
				if (bufferedReader != null) {
					bufferedReader.close();
				}
			} catch (IOException e) {}
		}
		return professionMapping;
	}

}
