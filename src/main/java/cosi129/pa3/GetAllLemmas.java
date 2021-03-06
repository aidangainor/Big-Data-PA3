package cosi129.pa3;

import java.io.IOException;

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

public class GetAllLemmas {
	public static class LemmaIndexReader extends Mapper<Text, Text, Text, IntWritable> { 
		public void map(Text articleId, Text indices, Context context) 
				throws IOException, InterruptedException {
			StringIntegerList wordFreqList = new StringIntegerList();
			wordFreqList.readFromString(indices.toString());
			for (StringInteger index : wordFreqList.getIndices()) {
				String lemma = index.getString().replaceAll("[^\\p{Alnum}\\s]", "");
				if (lemma.length() > 1) {
					context.write(new Text(index.getString()), new IntWritable(1));
				}
			}
		}
	}
	
	public static class WriteWord extends Reducer<Text, IntWritable, Text, Text> {	
		public void reduce(Text word, Iterable<IntWritable> values, Context context) 
				throws IOException, InterruptedException {
	    	int sum = 0;
			for (IntWritable val : values) {
				sum += val.get();
			}
			
			if (sum > 5) {
				context.write(word, new Text(""));
			}
		}
	}
	
	public static void main(String[] args) 
			throws IOException, ClassNotFoundException, InterruptedException {
		Configuration conf = new Configuration();
		String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
		
		Job job = Job.getInstance(conf);
		job.setJarByClass(GetAllLemmas.class);
		job.setMapperClass(LemmaIndexReader.class); 
		job.setReducerClass(WriteWord.class);
		
		job.setInputFormatClass(KeyValueTextInputFormat.class);
		
		conf.set("mapreduce.input.keyvaluelinerecordreader.key.value.separator","\t");
		job.setMapOutputValueClass(IntWritable.class);
		job.setMapOutputKeyClass(Text.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		
		try {
			FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
			FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));
			System.exit(job.waitForCompletion(true)? 0: 1);
		} catch (ArrayIndexOutOfBoundsException e) {
			System.out.println("Usage: GetAllLeamms input_dir output_dir");
		}
	}
}