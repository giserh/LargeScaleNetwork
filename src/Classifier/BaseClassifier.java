package Classifier;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;

import structures._Corpus;
import structures._Doc;
import utils.Utils;


public abstract class BaseClassifier {
	protected int m_classNo; //The total number of classes.
	protected int m_featureSize;
	protected _Corpus m_corpus;
	protected ArrayList<_Doc> m_trainSet; //All the documents used as the training set.
	protected ArrayList<_Doc> m_testSet; //All the documents used as the testing set.
	protected double[] m_cProbs;
	int i = 0;
	//for cross-validation
	protected int[][] m_confusionMat, m_TPTable;//confusion matrix over all folds, prediction table in each fold
	protected ArrayList<double[][]> m_precisionsRecalls; //Use this array to represent the precisions and recalls.

	protected String m_debugOutput; // set up debug output (default: no debug output)
	
	public void train() {
		train(m_trainSet);
	}
	
	public abstract void train(Collection<_Doc> trainSet);
	public abstract int predict(_Doc doc);
	protected abstract void init(); // to be called before training starts
	protected abstract void debug(_Doc d);
	
	public void test() throws FileNotFoundException{
//		i++;
//		PrintWriter writer = new PrintWriter(new File("./data/DebugOutput"+i+".dat"));
//		writer.write("DocID\t"+"True Label\t"+"Predicted Label\t\t"+"Content\n");
		for(_Doc doc: m_testSet){
			doc.setPredictLabel(predict(doc)); //Set the predict label according to the probability of different classes.
			int pred = doc.getPredictLabel(), ans = doc.getYLabel();
			m_TPTable[pred][ans] += 1; //Compare the predicted label and original label, construct the TPTable.
//			writer.write(doc.getID()+"\t"+doc.getYLabel()+"\t"+doc.getPredictLabel()+"\t"+doc.getSource()+"\n");
			if (m_debugOutput!=null && pred != ans)
				debug(doc);
		}
//		writer.close();
		m_precisionsRecalls.add(calculatePreRec(m_TPTable));
	}
	
	// Constructor with parameters.
	public BaseClassifier(_Corpus c, int class_number, int featureSize) {
		m_classNo = class_number;
		m_featureSize = featureSize;
		m_corpus = c;
		m_trainSet = new ArrayList<_Doc>();
		m_testSet = new ArrayList<_Doc>();
		m_cProbs = new double[m_classNo];
		m_TPTable = new int[m_classNo][m_classNo];
		m_confusionMat = new int[m_classNo][m_classNo];
		m_precisionsRecalls = new ArrayList<double[][]>();
		m_debugOutput = null;
	}
	
	public void setDebugOutput(String filename) {
		if (filename==null || filename.isEmpty())
			return;
		
		File f = new File(filename);
		if(!f.isDirectory()) { 
			if (f.exists()) 
				f.delete();
			m_debugOutput = filename;
		} else 
			System.err.println("Please specify a correct path for debug output!");
	}
	
	//k-fold Cross Validation.
	public void crossValidation(int k, _Corpus c) throws FileNotFoundException{
		c.shuffle(k);
		int[] masks = c.getMasks();
		ArrayList<_Doc> docs = c.getCollection();
		//Use this loop to iterate all the ten folders, set the train set and test set.
		for (int i = 0; i < k; i++) {
			for (int j = 0; j < masks.length; j++) {
				if( masks[j]==i ) 
					m_testSet.add(docs.get(j));
				else 
					m_trainSet.add(docs.get(j));
			}
			long start = System.currentTimeMillis();
			train();
			test();
			System.out.format("%s Train/Test finished in %.2f seconds.\n", this.toString(), (System.currentTimeMillis()-start)/1000.0);
			m_trainSet.clear();
			m_testSet.clear();
		}
		calculateMeanVariance(m_precisionsRecalls);	
	}
	
	//Split the data set as k folder, but only try one folder.
	public void crossValidation2(int k, _Corpus c) throws FileNotFoundException {
		c.shuffle(k);
		int[] masks = c.getMasks();
		ArrayList<_Doc> docs = c.getCollection();
		// We only need one folder, so no loop is needed any more.
		int i = 0;
		for (int j = 0; j < masks.length; j++) {
			if (masks[j] == i)
				m_testSet.add(docs.get(j));
			else
				m_trainSet.add(docs.get(j));
		}
		long start = System.currentTimeMillis();
		train();
		test();
		System.out.format("%s Train/Test finished in %.2f seconds.\n", this.toString(), (System.currentTimeMillis() - start) / 1000.0);
		m_trainSet.clear();
		m_testSet.clear();
		
		System.out.println("---------------------------------------------------------------------");
		System.out.format("The final result is as follows: The total number of classes is %d.\n", m_classNo);
		printConfusionMat();
		System.out.println("---------------------------------------------------------------------");
	}

	//Cross Validation to verify if the order matters in classifers. We need the start point and end point to split the trainging and testing dataset.
	public void crossValidation3(double sp, double ep, _Corpus c) throws FileNotFoundException {
		m_precisionsRecalls.clear(); // Clear previous precisions and recalls first.
		m_confusionMat = new int[m_classNo][m_classNo];	//Clear the m_confusionMat. Is this a efficient way to clear?	
		ArrayList<_Doc> docs = c.getCollection();
		Collections.sort(docs, new Comparator<_Doc>(){
			public int compare(_Doc d1, _Doc d2){
				if (d1.getTimeStamp() > d2.getTimeStamp()) return 1;
				else if (d1.getTimeStamp() < d2.getTimeStamp())return -1;
				else return 0;
			}
		});
		// Use this loop to iterate all the ten folders, set the train set and the test set.
		int totalSize = docs.size();
		int startTrain = (int) (sp * totalSize);
		int endTrain = (int) (ep * totalSize);
		for (int i = 0; i < totalSize; i++) {
			if(i >= startTrain && i < endTrain)
				m_trainSet.add(docs.get(i));
			else
				m_testSet.add(docs.get(i));
		}
		long start = System.currentTimeMillis();
		train();
		test();
		System.out.format("%s Train/Test finished in %.2f seconds...\n", this.toString(), (System.currentTimeMillis() - start) / 1000.0);
		m_trainSet.clear();
		m_testSet.clear();
		//In ttest, there is only one folder, so we only get the average precision, average recall, average F.
	}
	
	abstract public void saveModel(String modelLocation);
	
	//Calculate the precision and recall for one folder tests.
	public double[][] calculatePreRec(int[][] tpTable) {
//		double PreSum = 0, RecSum = 0, FSum = 0;//ttest
		double[][] PreRecOfOneFold = new double[m_classNo][2];
		for (int i = 0; i < m_classNo; i++) {
			PreRecOfOneFold[i][0] = (double) tpTable[i][i] / (Utils.sumOfRow(tpTable, i) + 0.001);// Precision of the class.
			PreRecOfOneFold[i][1] = (double) tpTable[i][i] / (Utils.sumOfColumn(tpTable, i) + 0.001);// Recall of the class.
//			PreSum += PreRecOfOneFold[i][0];
//			RecSum += PreRecOfOneFold[i][1];
			
			for(int j=0; j< m_classNo; j++) {
				m_confusionMat[i][j] += tpTable[i][j];
				tpTable[i][j] = 0; // clear the result in each fold
			}
		}
//		PreSum = PreSum / 5;
//		RecSum = RecSum / 5;
//		FSum = 2 * PreSum * RecSum / (PreSum + RecSum);
//		System.out.format("precision %.3f recall %.3f F1 %.3f\n", PreSum, RecSum, FSum);
		return PreRecOfOneFold;
	}
	
	public void printConfusionMat() {
		for(int i=0; i<m_classNo; i++)
			System.out.format("\t%d", i);
		
		double total = 0, correct = 0;
		double[] columnSum = new double[m_classNo], prec = new double[m_classNo];
		System.out.println("\tP");
		for(int i=0; i<m_classNo; i++){
			System.out.format("%d", i);
			double sum = 0; // row sum
			for(int j=0; j<m_classNo; j++) {
				System.out.format("\t%d", m_confusionMat[i][j]);
				sum += m_confusionMat[i][j];
				columnSum[j] += m_confusionMat[i][j];
				total += m_confusionMat[i][j];
			}
			correct += m_confusionMat[i][i];
			prec[i] = m_confusionMat[i][i]/(sum + 0.001); 
			System.out.format("\t%.4f\n", prec[i]);
		}
		
		System.out.print("R");
		for(int i=0; i<m_classNo; i++){
			columnSum[i] = m_confusionMat[i][i]/(columnSum[i] + 0.001); // recall
			System.out.format("\t%.4f", columnSum[i]);
		}
		System.out.format("\t%.4f", correct/total);
		
		System.out.print("\nF1");
		for(int i=0; i<m_classNo; i++)
			System.out.format("\t%.4f", 2.0 * columnSum[i] * prec[i] / (columnSum[i] + prec[i] + 0.001));
		System.out.println();
	}
	
	//Calculate the mean and variance of precision and recall.
	public double[][] calculateMeanVariance(ArrayList<double[][]> prs){
		//Use the two-dimension array to represent the final result.
		double[][] metrix = new double[m_classNo][4]; 
			
		double precisionSum = 0.0;
		double precisionVarSum = 0.0;
		double recallSum = 0.0;
		double recallVarSum = 0.0;

		//i represents the class label, calculate the mean and variance of different classes.
		for(int i = 0; i < m_classNo; i++){
			precisionSum = 0;
			recallSum = 0;
			// Calculate the sum of precisions and recalls.
			for (int j = 0; j < prs.size(); j++) {
				precisionSum += prs.get(j)[i][0];
				recallSum += prs.get(j)[i][1];
			}
			
			// Calculate the means of precisions and recalls.
			metrix[i][0] = precisionSum/prs.size();
			metrix[i][1] = recallSum/prs.size();
		}

		// Calculate the sum of variances of precisions and recalls.
		for (int i = 0; i < m_classNo; i++) {
			precisionVarSum = 0.0;
			recallVarSum = 0.0;
			// Calculate the sum of precision variance and recall variance.
			for (int j = 0; j < prs.size(); j++) {
				precisionVarSum += (prs.get(j)[i][0] - metrix[i][0])*(prs.get(j)[i][0] - metrix[i][0]);
				recallVarSum += (prs.get(j)[i][1] - metrix[i][1])*(prs.get(j)[i][1] - metrix[i][1]);
			}
			
			// Calculate the means of precisions and recalls.
			metrix[i][2] = Math.sqrt(precisionVarSum/prs.size());
			metrix[i][3] = Math.sqrt(recallVarSum/prs.size());
		}
		
		// The final output of the computation.
		System.out.println("---------------------------------------------------------------------");
		System.out.format("The final result is as follows: The total number of classes is %d.\n", m_classNo);
		
		for(int i = 0; i < m_classNo; i++)
			System.out.format("Class %d:\tprecision(%.3f+/-%.3f)\trecall(%.3f+/-%.3f)\n", i, metrix[i][0], metrix[i][2], metrix[i][1], metrix[i][3]);
		
		printConfusionMat();
		System.out.println("---------------------------------------------------------------------");
		return metrix;
	}
}
