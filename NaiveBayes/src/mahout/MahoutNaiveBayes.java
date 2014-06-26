package mahout;

import java.io.BufferedReader;

import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.Writer;
import org.apache.hadoop.io.Text;
import org.apache.mahout.classifier.naivebayes.NaiveBayesModel;
import org.apache.mahout.classifier.naivebayes.StandardNaiveBayesClassifier;
import org.apache.mahout.classifier.naivebayes.test.TestNaiveBayesDriver;
import org.apache.mahout.classifier.naivebayes.training.TrainNaiveBayesJob;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class MahoutNaiveBayes {
	public static void main(String [] args) throws Exception{
		String trainDataFile = "/home/supun/workspace/Mahout/data/train2.csv";
		String testDataFile = "/home/supun/workspace/Mahout/data/train2.csv";
		String validateDataFile = "/home/supun/workspace/Mahout/data/validate.csv";
		String seqPath = "/home/supun/workspace/Mahout/resources/sequence/";
		String modelPath = "/home/supun/workspace/Mahout/resources/model";
		String labelIndexPath = "/home/supun/workspace/Mahout/resources/labelIndex";
		String validateResultsPath = "/home/supun/workspace/Mahout/resources/results";
		Configuration configuration = new Configuration();
		
		//create train sequence
		createSequenceFile(trainDataFile,seqPath+"trainSeq",configuration);
		
		//create validate sequence
		createSequenceFile(validateDataFile,seqPath+"validateSeq",configuration);
		
		//train
		TrainNaiveBayesJob trainer = new TrainNaiveBayesJob();
		trainer.setConf(configuration);
		String[] trainingParameters = {"-i",seqPath+"trainSeq", "-o",modelPath,"-li",labelIndexPath,"-ow", "-el"};
		trainer.run(trainingParameters);
		
		//validate
		TestNaiveBayesDriver test =new TestNaiveBayesDriver();
		test.setConf(configuration);
		String[] testingParameters = {"-i",seqPath+"validateSeq", "-m",modelPath,"-l",labelIndexPath,"-o",validateResultsPath,"-ow",};
		test.run(testingParameters);
		
		// test (predict)
		NaiveBayesModel model;
        try {
	        model = NaiveBayesModel.materialize(new Path(modelPath), configuration);
			StandardNaiveBayesClassifier classifier = new StandardNaiveBayesClassifier(model);
			Vector testDataVector=createDataVector(testDataFile);
			System.out.println("Input: "+testDataVector);
			System.out.println("Prediction: "+classifier.classifyFull(testDataVector));
        } catch (IOException e) {
	        e.printStackTrace();
        }
	}
	
	public static void createSequenceFile(String inputData,String seqPath,Configuration configuration) throws IOException {
		FileSystem fs = FileSystem.get(configuration);
		Writer writer = new SequenceFile.Writer(fs, configuration, new Path(seqPath), Text.class, VectorWritable.class);
		Text key = new Text();
		VectorWritable vector = new VectorWritable();		
		BufferedReader reader = new BufferedReader(new FileReader(inputData));
		int count = 0;
		String line;
		
		/*
		 * find the number of features 
		 *  also, ignore the first line (headers) of data
		 */
		int featureCount=reader.readLine().split(",").length;
		double [] tmp = new double[featureCount];
		Vector tmpVector=new RandomAccessSparseVector(featureCount);
		
		/*
		 * create the vector and write to the sequence file
		 */
		while((line = reader.readLine()) != null) {
			String[] values = line.split(",");
			// set response variable class as key			
			key.set("/"+values[0]+"/"+values[0]);

			//create a data array for a given row
    		for(int i=1 ; i<values.length ; i++){
    			tmp[i]=(Double.parseDouble(values[i]));
    		}
    		//assign it to the vector
    		tmpVector.assign(tmp);
			vector.set(tmpVector);
   			writer.append(key, vector);
			count++;
		}
		reader.close();
		writer.close();		
		System.out.println("Wrote " + count + " entries.");
	}
	
	public static Vector createDataVector(String inputData) throws IOException{
		BufferedReader reader = new BufferedReader(new FileReader(inputData));
		String line;		
		//ignore the first line (headers)
		reader.readLine();
		
		while((line = reader.readLine()) != null) {
			String[] values = line.split(",");
			Vector tmpVector=new RandomAccessSparseVector(values.length);
			double [] tmp = new double[values.length];
    		for(int i=1 ; i<values.length ; i++){
    			tmp[i]=(Double.parseDouble(values[i]));
    		}
    		tmpVector.assign(tmp);
    		reader.close();
    		return tmpVector;
		}
		reader.close();
		return null;
	}
}
