package cosi129.pa3;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.HashMap;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class CreateTestData {
	// Read the small spit of names used in 
	private final static String PROFESSIONS_TEST_FILE = "professions_test.txt";
	
	public static class TestDataMapper extends Mapper<Text, Text, Text, Text> {
        private HashMap<String, ArrayList<String>> professionMapping;
        
        // Setup builds hash map of names-to-professions before mapper jobs
        protected void setup(Context context) throws IOException, InterruptedException {
        	professionMapping = LemmaIndexFormater.getProfessionMapping(PROFESSIONS_TEST_FILE);
    	}
        
		public void map(Text articleId, Text indices, Context context) 
				throws IOException, InterruptedException {
			// Check if this person has profession classified in test set split, if not ignore
			String articleIdTrimmed = articleId.toString().trim();
			if (professionMapping.containsKey(articleIdTrimmed)) {
				context.write(articleId, indices);
			}
		}
	}
	
	public static void main(String[] args) 
			throws IOException, ClassNotFoundException, InterruptedException, URISyntaxException {
		Configuration conf = new Configuration();
		String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
		Job job = Job.getInstance(conf);
		job.setJarByClass(CreateTestData.class);
		job.setMapperClass(TestDataMapper.class); 
		job.setNumReduceTasks(0);
		job.setInputFormatClass(KeyValueTextInputFormat.class);
		
		conf.set("mapreduce.input.keyvaluelinerecordreader.key.value.separator","\t");
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		
		FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
		FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));
		System.exit(job.waitForCompletion(true)? 0: 1);	
	}
}
