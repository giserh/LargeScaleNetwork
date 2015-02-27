package Classifier;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Random;
import structures.MyPriorityQueue;
import structures._Corpus;
import structures._Doc;
import structures._RankItem;
import utils.Utils;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2D;

public class SemiSupervised extends BaseClassifier{
	protected double[] m_pY;//p(Y), the probabilities of different classes.
	protected double m_labelRatio; // percentage of training data for semi-supervised learning
	protected double m_TLalpha; //Weight coefficient between unlabeled node and labeled node.
	protected double m_TLbeta; //Weight coefficient between unlabeled node and unlabeled node.
	protected double m_M; //Influence of labeled node.
	protected int m_k; // k labeled nodes.
	protected int m_kPrime;//k' unlabeled nodes.
	protected double m_discount = 0.5; // default similarity discount if across different products
	protected double[][] m_QQplot;
	
	private int m_U, m_L; 
	private double[] m_Y; //The predicted labels for both labeled and unlabeled data.
	private double[] m_Y_U;//The predicted labels for unlabeled data.
	/**For debugging purpose**/
	private String[][] m_debugOutput;//The two dimension array is used to represent the debug output for all unlabeled data.
	private int[][] m_Top5UnlabeledIndex; //The top5 unlabeled data's index.
	private int[][] m_neighborUnlabeled; 
	private double[][] m_similarities;
	
	private double[] m_cache; // cache the similarity computation results given the similarity metric is symmetric
	protected MyPriorityQueue<_RankItem> m_kUL, m_kUU; // k nearest neighbors for Unlabeled-Labeled and Unlabeled-Unlabeled
	protected ArrayList<_Doc> m_labeled; // a subset of training set
	
	protected BaseClassifier m_classifier; //Multiple learner.	
	protected PrintWriter m_writer;
	
	//Default constructor without any default parameters.
	public SemiSupervised(_Corpus c, int classNumber, int featureSize, String classifier){
		super(c, classNumber, featureSize);
		m_pY = new double[m_classNo];
		m_labelRatio = 0.1;
		m_TLalpha = 1.0;
		m_TLbeta = 0.1;
		m_M = 10000;
		m_k = 100;
		m_kPrime = 50;	
		m_labeled = new ArrayList<_Doc>();
		setClassifier(classifier);
	}	
	
	//Constructor: given k and kPrime
	public SemiSupervised(_Corpus c, int classNumber, int featureSize, String classifier, int k, int kPrime){
		super(c, classNumber, featureSize);
		m_pY = new double[m_classNo];
		m_labelRatio = 0.1;
		m_TLalpha = 1.0;
		m_TLbeta = 0.1;
		m_M = 10000;
		m_k = k;
		m_kPrime = kPrime;	
		m_labeled = new ArrayList<_Doc>();
		setClassifier(classifier);
	}
	
	//Constructor: given k, kPrime, TLalpha and TLbeta
	public SemiSupervised(_Corpus c, int classNumber, int featureSize, String classifier, int k, int kPrime, double TLalhpa, double TLbeta){
		super(c, classNumber, featureSize);
		m_pY = new double[m_classNo];
		m_labelRatio = 0.1;
		m_TLalpha = TLalhpa;
		m_TLbeta = TLbeta;
		m_M = 10000;
		m_k = k;
		m_kPrime = kPrime;	
		m_labeled = new ArrayList<_Doc>();
		setClassifier(classifier);
	}
	
	public SemiSupervised(_Corpus c, int classNumber, int featureSize, String classifier, double ratio, int k, int kPrime, double TLalhpa, double TLbeta){
		super(c, classNumber, featureSize);
		m_pY = new double[m_classNo];
		m_labelRatio = ratio;
		m_TLalpha = TLalhpa;
		m_TLbeta = TLbeta;
		m_M = 10000;
		m_k = k;
		m_kPrime = kPrime;	
		m_labeled = new ArrayList<_Doc>();
		setClassifier(classifier);
	}
	@Override
	public String toString() {
		return String.format("Transductive Learning[C:%s, k:%d, k':%d]", m_classifier, m_k, m_kPrime);
	}
	
	private void setClassifier(String classifier) {
		if (classifier.equals("NB"))
			m_classifier = new NaiveBayes(null, m_classNo, m_featureSize);
		else if (classifier.equals("LR"))
			m_classifier = new LogisticRegression(null, m_classNo, m_featureSize);
		else if (classifier.equals("SVM"))
			m_classifier = new SVM(null, m_classNo, m_featureSize);
		else {
			System.out.println("Classifier has not developed yet!");
			System.exit(-1);
		}
	}
	
	@Override
	protected void init() {
		m_labeled.clear();
		Arrays.fill(m_pY, 0);
	}
	
	//Train the data set.
	public void train(Collection<_Doc> trainSet){
		//m_classifier.train(trainSet);//Multiple learner.		
		init();

		//Randomly pick some training documents as the labeled documents.
		Random r = new Random();
		for (_Doc doc: trainSet){
			m_pY[doc.getYLabel()]++;
			if(r.nextDouble()<m_labelRatio){
				m_labeled.add(doc);
			}
		}
	}
	
	private void initCache() {
		m_cache = new double[m_U*(2*m_L+m_U-1)/2];//specialized for the current matrix structure
	}
	
	private int encode(int i, int j) {
		if (i>j) {//swap
			int t = i;
			i = j;
			j = t;
		}
		return (2*(m_U+m_L-1)-i)/2*(i+1) - ((m_U+m_L)-j);//specialized for the current matrix structure
	}
	
	private void setCache(int i, int j, double v) {
		m_cache[encode(i,j)] = v;
	}
	
	private double getCache(int i, int j) {
		return m_cache[encode(i,j)];
	}
	
	//Test the data set, including the transductive learning process.
	public void test() throws FileNotFoundException{
		double similarity = 0, average = 0, sd = 0;
		m_L = m_labeled.size();
		m_U = m_testSet.size();
		m_Y = new double[m_U + m_L];
		m_Y_U = new double[m_U];
		m_debugOutput = new String[m_U][14];
		m_Top5UnlabeledIndex = new int[m_U][5];
		m_neighborUnlabeled = new int[5][5];
		m_similarities = new double[50][2];	
		m_QQplot = new double[m_U][4];
		
		double[] simi_U = new double[m_U];
		double[] simi_L = new double[m_L];
		/*** Set up cache structure for efficient computation. ****/
		initCache();

		/*** pre-compute the full similarity matrix (except the diagonal). ****/
		_Doc di, dj;
		for (int i = 0; i < m_U; i++) {
			Arrays.fill(simi_U, 0);
			Arrays.fill(simi_L, 0);
			di = m_testSet.get(i);
			
			for (int j = i + 1; j < m_U; j++) {// to save computation since our similarity metric is symmetric
				dj = m_testSet.get(j);
				similarity = Utils.calculateSimilarity(di, dj) * di.getWeight() * dj.getWeight();
				if (!di.sameProduct(dj))
					similarity *= m_discount;// differentiate reviews from different products
				setCache(i, j, similarity);
				setCache(j, i, similarity);
			}
			for(int k = 0; k < m_U; k++){
				if (k != i){
					if(k > i)
						simi_U[k] = getCache(i, k);
					else simi_U[k] = getCache(k, i);
				}
			}
			average = Utils.averageOfArray(simi_U);
			sd = Utils.sdOfArray(simi_U, average);
			m_QQplot[i][0] = average;
			m_QQplot[i][1] = sd;
			
			for (int j = 0; j < m_L; j++) {
				dj = m_labeled.get(j);
				similarity = Utils.calculateSimilarity(di, dj) * di.getWeight() * dj.getWeight();
				if (!di.sameProduct(m_labeled.get(j)))
					similarity *= m_discount;// differentiate reviews from different products
				setCache(i, m_U + j, similarity);
				setCache(m_U + j, i, similarity);
			}
			for(int k = m_U; k < m_U + m_L; k++){
					simi_L[k-m_U] = getCache(i, k);
			}
			average = Utils.averageOfArray(simi_U);
			sd = Utils.sdOfArray(simi_U, average);
			m_QQplot[i][2] = average;
			m_QQplot[i][3] = sd;
		}
		SparseDoubleMatrix2D mat = new SparseDoubleMatrix2D(m_U + m_L, m_U	+ m_L);
		/*** Set up structure for k nearest neighbors. ****/
		m_kUU = new MyPriorityQueue<_RankItem>(m_kPrime);
		m_kUL = new MyPriorityQueue<_RankItem>(m_k);
		/**** Construct the C+scale*\Delta matrix and Y vector. ****/
		double scale = -m_TLalpha / (m_k + m_TLbeta * m_kPrime), sum, value;
		for (int i = 0; i < m_U; i++) {
			//The basic info for the unlabeled data.
			m_debugOutput[i][0] = m_testSet.get(i).getItemID();
			m_debugOutput[i][1] = m_testSet.get(i).getSource();
			m_debugOutput[i][2] = Integer.toString(m_testSet.get(i).getYLabel());

			// set the part of unlabeled nodes. U-U
			for (int j = 0; j < m_U; j++) {
				if (j == i)
					continue;
				m_kUU.add(new _RankItem(j, getCache(i, j)));
			}
			sum = 0;
			for (_RankItem n : m_kUU) {
				value = Math.max(m_TLbeta * n.m_value, mat.getQuick(i, n.m_index) / scale);// recover the original Wij
				mat.setQuick(i, n.m_index, scale * value);
				mat.setQuick(n.m_index, i, scale * value);
				sum += value;
				m_neighborUnlabeled[m_testSet.get(i).getYLabel()][m_testSet.get(n.m_index).getYLabel()]++;//debug purpose.
			}
			//Put the top 5 unlabeled nearest data into the matrix.
			int topK = 0;
			while(topK < 50){
				_RankItem n = m_kUU.elementAt(topK);
				_Doc d = m_testSet.get(n.m_index);
				//m_Top5UnlabeledIndex[i][topK] = n.m_index;
				//m_debugOutput[i][topK+3]= Integer.toString(d.getYLabel())+"\t"+Double.toString(n.m_value);
				m_similarities[topK][0] = d.getYLabel();
				m_similarities[topK][1] = n.m_value;
				topK++;
			}
			outputSimilarities(i);
			m_kUU.clear();

			// Set the part of labeled and unlabeled nodes. L-U and U-L
			for (int j = 0; j < m_L; j++)
				m_kUL.add(new _RankItem(m_U + j, getCache(i, m_U + j)));
			for (_RankItem n : m_kUL) {
				value = Math.max(n.m_value, mat.getQuick(i, n.m_index) / scale);// recover the original Wij
				mat.setQuick(i, n.m_index, scale * value);
				mat.setQuick(n.m_index, i, scale * value);
				sum += value;
				m_neighborUnlabeled[m_testSet.get(i).getYLabel()][m_trainSet.get(n.m_index).getYLabel()]++;
			}
			mat.setQuick(i, i, 1 - scale * sum);
//			topK = 0;
//			while(topK < 5){
//				_RankItem n = m_kUL.elementAt(topK);
//				_Doc d = m_trainSet.get((n.m_index-m_U));
//				m_debugOutput[i][topK+8]= Integer.toString(d.getYLabel())+"\t"+Double.toString(n.m_value);
//				topK++;
//			}
			m_kUL.clear();

			// set up the Y vector for unlabeled data
			m_Y[i] = 0;
			//m_classifier.predict(m_testSet.get(i)); // Multiple learner.
		}
		for(int i=m_U; i<m_L+m_U; i++) {
			sum = 0;
			for(int j=0; j<m_U; j++) 
				sum += mat.getQuick(i, j);
			mat.setQuick(i, i, m_M-sum); // scale has been already applied in each cell
			//set up the Y vector for labeled data
			m_Y[i] = m_M * m_labeled.get(i-m_U).getYLabel();
		}
		/***Perform matrix inverse.****/
		DenseDoubleAlgebra alg = new DenseDoubleAlgebra();
		DoubleMatrix2D result = alg.inverse(mat);
		/*******Show results*********/
		//Collect all the labels for unlabeled data.
		for(int i = 0; i < m_U; i++){
			double pred = 0;
			for(int j=0; j<m_U+m_L; j++)
				pred += result.getQuick(i, j) * m_Y[j];
			m_Y_U[i] = pred;
		}
		//Set the predicted label according to threshold.
		for(int i = 0; i < m_Y_U.length; i++){
			m_TPTable[getLabel1(m_Y_U[i])][m_testSet.get(i).getYLabel()] += 1;
			m_debugOutput[i][13] = Integer.toString(getLabel3(m_Y_U[i]));
		}
		m_precisionsRecalls.add(calculatePreRec(m_TPTable));
		//debugOutput();
		//outputNeighbors();
	}
	
	//Print out the avg and sd of wij of unlabeled data and labeled data.
	public void printQQPlot() throws FileNotFoundException{	
		PrintWriter writer = new PrintWriter(new File("./data/QQplot.dat"));
		writer.print("Unlabeled\tU_avg\tU_sd\tL_avg\tL_sd");
		writer.println();
		for(int i = 0; i < m_U; i++){
			writer.print("U["+i+"]\t"+m_QQplot[i][0]+"\t"+m_QQplot[i][1]+"\t"+m_QQplot[i][2]+"\t"+m_QQplot[i][3]+"\n");
		}
		writer.close();
	}
	
	//Print out the debug info, the true label, the predicted label, the content. Its k and k' neighbors.
	public void debugOutput() throws FileNotFoundException{
		PrintWriter writer = new PrintWriter(new File("./data/debug/SemiSupervised.output"));
		for(int i = 0; i < m_debugOutput.length; i++){
			if(!m_debugOutput[i][2].equals(m_debugOutput[i][13])){
				writer.write(String.format("ProdID: %s\nContent: %s\nTrue Label: %s\tPredicted Label: %s\n", m_debugOutput[i][0], m_debugOutput[i][1], m_debugOutput[i][2], m_debugOutput[i][13]));
				writer.write("Top5 unlabeled data:\n");
				for(int j = 0; j < 5; j++)
					writer.write(String.format("True Label + Similarity: %s\tPredicted Label: %d\n", m_debugOutput[i][3+j], getLabel3(m_Y_U[m_Top5UnlabeledIndex[i][j]])));
				writer.write("Top5 labeled data:\n");
				for(int j = 0; j < 5; j++)
					writer.write(String.format("True Label + Similarity: %s\n", m_debugOutput[i][8+j]));
				writer.write("\n");
			}
		}
		writer.close();
	}
	
	//Print out the debug info, the true label, the predicted label, the content. Its k and k' neighbors.
	public void outputNeighbors() throws FileNotFoundException{
//		PrintWriter writer = new PrintWriter(new File("./data/Neighbors.dat"));
		for(int i = 0; i < 5; i++){
			for(int j = 0;  j < 5; j++){
				System.out.print(m_neighborUnlabeled[i][j] + "\t");
			}
			System.out.println();
			//writer.write("\n");
		}
		System.out.println("---------------------------------------------");
		//writer.close();
	}
	
	public void outputSimilarities(int i) throws FileNotFoundException {
		PrintWriter writer = new PrintWriter(new File("./data/similarity/Similairty"+i+".dat"));
		writer.write("True Label: "+ m_Y_U[i]+"\n");
		for(int j = 0; j < m_similarities.length; j++){
			writer.write(m_similarities[j][0]+"\t"+m_similarities[j][1]+"\n");
		}
		writer.close();
	}
	
	/**Different getLabel methods.**/
	//This is the original getLabel: -|c-p(c)|
	private int getLabel1(double pred) {
		for(int i=0; i<m_classNo; i++)
			m_cProbs[i] = -Math.abs(i-pred); //-|c-p(c)|
		return Utils.maxOfArrayIndex(m_cProbs);
	}
	//This is the second getLabel: |c-p(c)|
	private int getLabel2(double pred) {
		for(int i=0; i<m_classNo; i++)
			m_cProbs[i] = Math.abs(i-pred); //|c-p(c)|
		return Utils.minOfArrayIndex(m_cProbs);
	}
	//p(c) * exp(-|c-f(u_i)|)/sum_j{exp(-|c-f(u_j))} j represents all unlabeled data
	private int getLabel3(double pred){
		double sum = 0;
		// Calculate the probabilities of different classes.
		for (int i = 0; i < m_classNo; i++) {
			sum += m_pY[i];
		}
		for (int i = 0; i < m_classNo; i++) {
			m_pY[i] = m_pY[i] / sum;
		}
		//Calculate the denominator first.
		double[] denominators = new double[m_classNo];
		for(int i = 0; i < m_classNo; i++){
			for(int j = 0; j < m_U; j++){
				denominators[i] += Math.exp(-Math.abs(i - m_Y_U[j]));
			}
			m_cProbs[i] = m_pY[i] * Math.exp(-Math.abs(i-pred)) / denominators[i];
		}
		return Utils.maxOfArrayIndex(m_cProbs);
	}
	//exp(-|c-f(u_i)|)/sum_j{exp(-|c-f(u_j))} j represents all unlabeled data, without class probabilities.
	private int getLabel4(double pred) {
		double sum = 0;
		// Calculate the probabilities of different classes.
		for (int i = 0; i < m_classNo; i++) {
			sum += m_pY[i];
		}
		for (int i = 0; i < m_classNo; i++) {
			m_pY[i] = m_pY[i] / sum;
		}
		// Calculate the denominator first.
		double[] denominators = new double[m_classNo];
		for (int i = 0; i < m_classNo; i++) {
			for (int j = 0; j < m_U; j++) {
				denominators[i] += Math.exp(-Math.abs(i - m_Y_U[j]));
			}
			m_cProbs[i] = Math.exp(-Math.abs(i - pred)) / denominators[i];
		}
		return Utils.maxOfArrayIndex(m_cProbs);
	}
	
	@Override
	protected void debug(_Doc d){} // no easy way to debug
	
	@Override
	public int predict(_Doc doc) {
		return -1; //we don't support this
	}
	
	//Save the parameters for classification.
	@Override
	public void saveModel(String modelLocation){
		
	}
}
