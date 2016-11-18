package cosi129.pa3;

import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Scanner;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.ArrayList;

import cosi129.pa3.StringIntegerList;
import cosi129.pa3.StringIntegerList.StringInteger;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.Writer;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;

public class NaiveBayes {
	public static class ProfessionMapper extends Mapper<Text, Text, Text, Text> {
        private final static String PROFESSIONS_FILE = "professions.txt";
        private HashMap<String, List<String>> professionMapping;
        
        // Setup builds hash map of names-to-professions before mapper jobs
        protected void setup(Context context) throws IOException, InterruptedException {
            // Get professions file as input stream from jar resources
            InputStream in = getClass().getResourceAsStream(PROFESSIONS_FILE);
            Scanner sc = new Scanner(new InputStreamReader(in));
            professionMapping = new HashMap<String, List<String>>();
           
            // Scan through file line by line
            while (sc.hasNextLine()) {
                String line = sc.nextLine();
                String[] split = line.split(":");

                
                String name = "";
                // Separate names and their professions
                if(split.length > 0) name = split[0].trim();
                String[] professions = new String[1];
                if(split.length > 1 ) professions = split[1].split(","); 
                List<String> profList = new ArrayList<String>();
               
                for (String prof : professions) {
                    if(prof != null) profList.add(prof.trim());
                }
               
                professionMapping.put(name, profList);
            }
            sc.close();
        }
        
		public void map(Text articleId, Text indices, Context context) throws IOException, InterruptedException {
			// Check if this person has profession classified, if not ignore
			String articleIdTrimmed = articleId.toString().trim();
			if (professionMapping.containsKey(articleIdTrimmed)) {
				StringIntegerList wordFreqList = new StringIntegerList();
				wordFreqList.readFromString(indices.toString());
				List<String> professions = professionMapping.get(articleIdTrimmed);
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
			
			System.out.println("Profession = " + profession);
			
			Iterator<Text> iter = terms.iterator();
			Configuration conf = context.getConfiguration();
			String professionFormated = profession.toString().replaceAll(" ", "_");
			StringBuilder docLems = new StringBuilder();
			// Since the SequenceFile API does not support writing to existing file in Hadoop 2.3, we must make multiple .seq files
			// While normally we would write 1 big .seq files for a profession, heap space limitations prevent this so we must keep flushing to disk
			int seqFilesWrittenCount = 0;
			
			while(iter.hasNext()) {
				// Unroll the lemma and word count pairs so they can be formatted correctly for mahout
				// For example, math, 3 ==> math math math
				String si = iter.next().toString();
				String[] lemmaAndCount = si.split(",");
				String lemma = lemmaAndCount[0] + " ";
				int lemmaCount = 0;
				try {
					lemmaCount = Integer.parseInt(lemmaAndCount[1]);
				} catch (NumberFormatException e) {
					// This means malformed data from the provided lemma index
					// This is not our problem, so ignore this exception
				}
				
				for (int i = 0; i < lemmaCount; i++){
					// Set buffer size to 512, this basically means sequence files are maximum length of 512 to prevent heap overflow
					if (docLems.length() > 512) {
						// Append the lemma to our buffer, also remove whitespace at end since this is final lemma added
						// Flush our string buffer to HDFS as a sequence file, and then increment our file count for this profession
						docLems.append(lemma.trim());
						URI path = URI.create("seqfiles/" + professionFormated + seqFilesWrittenCount + ".seq");
						FileSystem fs = FileSystem.get(path, conf);
						SequenceFile.Writer writer = new Writer(fs, conf, new Path(path), profession.getClass(), Text.class);
						writer.append(profession, new Text(docLems.toString()));
						seqFilesWrittenCount++;
						docLems.setLength(0);
						writer.close();
					} else {
						docLems.append(lemma);
					}
				}
			}
		}
	}
	
	public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
		Configuration conf = new Configuration();
		String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
		
		Job job = Job.getInstance(conf);
		job.setJarByClass(NaiveBayes.class);
		job.setMapperClass(ProfessionMapper.class); 
		job.setReducerClass(ProfessionReducer.class);
		
		job.setInputFormatClass(KeyValueTextInputFormat.class);
		conf.set("mapreduce.input.keyvaluelinerecordreader.key.value.separator","\t");
		job.setMapOutputValueClass(Text.class);
		job.setMapOutputKeyClass(Text.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
		FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));
		System.exit(job.waitForCompletion(true)? 0: 1);
	}
}
