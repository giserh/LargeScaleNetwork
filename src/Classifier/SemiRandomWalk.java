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

public class SemiRandomWalk extends BaseClassifier{
	protected double[] m_pY;//p(Y), the probabilities of different classes.
	protected double m_labelRatio; // percentage of training data for semi-supervised learning
	protected double m_TLalpha; //Weight coefficient between unlabeled node and labeled node.
	protected double m_TLbeta; //Weight coefficient between unlabeled node and unlabeled node.
	protected double m_M; //Influence of labeled node.
	protected int m_k; // k labeled nodes.
	protected int m_kPrime;//k' unlabeled nodes.
	protected double m_discount = 0.5; // default similarity discount if across different products
	
	protected double m_difference; //The difference between the previous labels and current labels.
	protected double m_eta; //The parameter used in random. 
	protected ArrayList<_Doc> m_labeled; // a subset of training set
	protected BaseClassifier m_classifier; //Multiple learner.	
	
	protected int m_U, m_L; 
	protected double[] m_Y_U;//The predicted labels for unlabeled data.
	protected String[][] m_debugOutput;//The two dimension array is used to represent the debug output for all unlabeled data.
	protected double[] m_cache; // cache the similarity computation results given the similarity metric is symmetric
	protected MyPriorityQueue<_RankItem> m_kUL, m_kUU; // k nearest neighbors for Unlabeled-Labeled and Unlabeled-Unlabeled
		
	protected PrintWriter m_writer;
	protected int[] m_Pmul; //The predicted results of multiple learner for all files in test set.
	
	//Default constructor without any default parameters.
	public SemiRandomWalk(_Corpus c, int classNumber, int featureSize, String classifier){
		super(c, classNumber, featureSize);
		m_pY = new double[classNumber];
		m_labelRatio = 0.1;
		m_TLalpha = 1.0;
		m_TLbeta = 0.1;
		m_M = 10000;
		m_k = 100;
		m_kPrime = 50;	
		
		m_difference = 10000;
		m_eta = 0.1;
		m_labeled = new ArrayList<_Doc>();
		setClassifier(classifier);
	}	
	
	//Constructor: given k and kPrime
	public SemiRandomWalk(_Corpus c, int classNumber, int featureSize, String classifier, int k, int kPrime){
		super(c, classNumber, featureSize);
		m_pY = new double[classNumber];
		m_labelRatio = 0.1;
		m_TLalpha = 1.0;
		m_TLbeta = 0.1;
		m_M = 10000;
		m_k = k;
		m_kPrime = kPrime;	
		
		m_difference = 10000;
		m_eta = 0.1;
		m_labeled = new ArrayList<_Doc>();
		setClassifier(classifier);
	}
	
	public SemiRandomWalk(_Corpus c, int classNumber, int featureSize, String classifier, double ratio, int k, int kPrime, double TLalhpa, double TLbeta){
		super(c, classNumber, featureSize);
		m_pY = new double[classNumber];
		m_labelRatio = ratio;
		m_TLalpha = TLalhpa;
		m_TLbeta = TLbeta;
		m_M = 10000;
		m_k = k;
		m_kPrime = kPrime;	
		
		m_difference = 10000;
		m_eta = 0.1;
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
		m_classifier.train(trainSet);//Multiple learner.		
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
	
	//Preprocess to set the cache.
	public void RandomWalkPreprocess(){
		double similarity = 0;
		/*** Set up cache structure for efficient computation. ****/
		initCache();
		/*** pre-compute the full similarity matrix (except the diagonal). ****/
		_Doc di, dj;
		for (int i = 0; i < m_U; i++) {
			di = m_testSet.get(i);
			for (int j = i + 1; j < m_U; j++) {// to save computation since our similarity metric is symmetric
				dj = m_testSet.get(j);
				similarity = Utils.calculateSimilarity(di, dj) * di.getWeight() * dj.getWeight();
				if (!di.sameProduct(dj))
					similarity *= m_discount;// differentiate reviews from different products
				setCache(i, j, similarity);
			}

			for (int j = 0; j < m_L; j++) {
				dj = m_labeled.get(j);
				similarity = Utils.calculateSimilarity(di, dj) * di.getWeight() * dj.getWeight();
				if (!di.sameProduct(m_labeled.get(j)))
					similarity *= m_discount;// differentiate reviews from different products
				setCache(i, m_U + j, similarity);
			}
		}
	}
	
	//The random walk algorithm to generate new labels for unlabeled data.
	//Take the average of all neighbors as the new label until they converge.
	public double[] RandomWalk(double[] Y_U){
		double [] PreResults = new double[Y_U.length];
		for(int i = 0; i < Y_U.length; i++)
			PreResults[i] = Y_U[i];
		m_difference = 0;
		
		/*** Set up structure for k nearest neighbors. ****/
		m_kUU = new MyPriorityQueue<_RankItem>(m_kPrime);
		m_kUL = new MyPriorityQueue<_RankItem>(m_k);

		/**** Construct the C+scale*\Delta matrix and Y vector. ****/
		for (int i = 0; i < m_U; i++) {
			double wijSum = 0;
			double labelSum = 0;
			/****Construct the top k' unlabeled data for the current data.****/
			for (int j = 0; j < m_U; j++) {
				if (j == i)
					continue;
				m_kUU.add(new _RankItem(j, getCache(i, j)));
			}
			/****Construct the top k labeled data for the current data.****/
			for (int j = 0; j < m_L; j++)
				m_kUL.add(new _RankItem(m_U + j, getCache(i, m_U + j)));
			
			/****Get the sum of k'UU******/
			for(_RankItem n: m_kUU){
				wijSum += n.m_value; //get the similarity between two nodes.
				labelSum += n.m_value * Y_U[n.m_index];
			}
			m_kUU.clear();
			/****Get the sum of kUL******/
			for(_RankItem n: m_kUL){
				wijSum += n.m_value;
				labelSum += n.m_value * m_labeled.get(n.m_index - m_U).getYLabel();
			}
			m_kUL.clear();
			Y_U[i] = m_eta * labelSum / wijSum + (1-m_eta) * m_Pmul[i];
		}
		
		double [] AfterResults = Y_U;
		if (PreResults.length == AfterResults.length) {
			for(int i = 0; i < PreResults.length; i++){
				m_difference += (PreResults[i] - AfterResults[i]) * (PreResults[i] - AfterResults[i]);
			}
		}
		return Y_U;
	}
	
	//The test for random walk algorithm.
	public void test(){
		m_L = m_labeled.size();
		m_U = m_testSet.size();
		m_Y_U = new double[m_U];
		double[] Y_U = new double[m_U];
		m_Pmul = new int[m_U];
		m_difference = 10000;
		//Before random walk, predict all unlabeled data according to the multiple learner.
		for(int i = 0; i < m_testSet.size(); i++){
			m_Pmul[i] = m_classifier.predict(m_testSet.get(i));
		}
		//Set the cache which contains all similarities.
		RandomWalkPreprocess(); 
		//Initialize all the elements to be 0.
		Arrays.fill(Y_U, 0);
		while(Math.sqrt(m_difference) > 1e-4){
			Y_U = RandomWalk(Y_U);
		}
		m_Y_U = Y_U;
		for(int i = 0; i < Y_U.length; i++){
			m_TPTable[getLabel3(Y_U[i])][m_testSet.get(i).getYLabel()] += 1;
		}
		m_precisionsRecalls.add(calculatePreRec(m_TPTable));
	}
	
	//Print out the debug info, the true label, the predicted label, the content. Its k and k' neighbors.
	public void Debugoutput() throws FileNotFoundException{
		PrintWriter writer = new PrintWriter(new File("./data/DebugOutput.xls"));
		writer.print("U[i]\tTrueLabel\tPredictedLabel\tTop5UnlabeledData\t\t\t\t\tTop5LabeledData\t\t\t\t\tContent\n");
		for(int i = 0; i < m_debugOutput.length; i++){
			writer.write(i+"\t");
			for(int j = 0; j < 13; j++){
				writer.write(m_debugOutput[i][j]+"\t");
			}
			writer.write("\n");
		}
		writer.close();
	}
	
	//This is the original getLabel: -|c-p(c)|
	private int getLabel1(double pred) {
		for(int i=0; i<m_classNo; i++)
			m_cProbs[i] = -Math.abs(i-pred); //-|c-p(c)|
		return Utils.maxOfArrayIndex(m_cProbs);
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
