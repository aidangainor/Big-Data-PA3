package cosi129.pa3;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.ArrayList;
import java.util.Scanner;

import cosi129.pa3.StringIntegerList.StringInteger;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class LemmaIndexFormater {
	private final static String PROFESSIONS_FILE = "professions.txt";
	
	/*
	 * From the professions.txt file, make a map of a person's name to their profession(s)
	 */
	public static HashMap<String, ArrayList<String>> getProfessionMapping() 
			throws IOException {
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
	
	public static class ProfessionMapper extends Mapper<Text, Text, Text, Text> {
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
					for (StringInteger index : wordFreqList.getIndices()) {
						String term = index.toString();
						context.write(new Text(profession), new Text(term));
					}
				}
			}
		}
	}
	
	public static class ProfessionReducer extends Reducer<Text, Text, Text, Text> {
		public void reduce(Text profession, Iterable<Text> terms, Context context) 
				throws IOException, InterruptedException {
			System.out.println("in reducer!");
			HashMap<String, Integer> lemmaFrequency = new HashMap<String, Integer>();
			Iterator<Text> iter = terms.iterator();
			
			while(iter.hasNext()) {
				String si = iter.next().toString();
				String[] lemmaAndCount = si.split(",");
				String lemma = lemmaAndCount[0];
				int lemmaCount = 0;
				try {
					lemmaCount = Integer.parseInt(lemmaAndCount[1]);
					if (lemmaFrequency.containsKey(lemma)) {
						lemmaFrequency.put(lemma, lemmaFrequency.get(lemma) + lemmaCount);
					} else {
						lemmaFrequency.put(lemma, lemmaCount);
					}
				} catch (NumberFormatException e) {
					// This means malformed data from the provided lemma index
					// This is not our problem, so ignore this exception.
				}
			}
			
			// Make lemmaFrequency hash into TF (term frequency) form, so we can later compute TF-IDF
			// To do this, we must get a count of lemmas in this profession
			int count = 0;
			for (String key : lemmaFrequency.keySet())
				count += lemmaFrequency.get(key);
			for (String key : lemmaFrequency.keySet()) {
				// Normalize data, we stay in Integer format to keep compatibility with StringIntegerList class
				lemmaFrequency.put(key, (int) ((lemmaFrequency.get(key)/(double) count) * 100000.0));
			}
			StringIntegerList lemmaToWrite = new StringIntegerList(lemmaFrequency);
			context.write(profession, new Text(lemmaToWrite.toString()));
		}
	}
	
	public static void main(String[] args) 
			throws IOException, ClassNotFoundException, InterruptedException {
		Configuration conf = new Configuration();
		String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
		
		Job job = Job.getInstance(conf);
		job.setJarByClass(LemmaIndexFormater.class);
		job.setMapperClass(ProfessionMapper.class); 
		job.setReducerClass(ProfessionReducer.class);
		
		job.setInputFormatClass(KeyValueTextInputFormat.class);
		
		conf.set("mapreduce.input.keyvaluelinerecordreader.key.value.separator","\t");
		job.setMapOutputValueClass(Text.class);
		job.setMapOutputKeyClass(Text.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		
		try {
			FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
			FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));
			System.exit(job.waitForCompletion(true)? 0: 1);
		} catch (ArrayIndexOutOfBoundsException e) {
			System.out.println("Usage: LemmaIndexFormater input_dir output_dir");
		}
	}
}