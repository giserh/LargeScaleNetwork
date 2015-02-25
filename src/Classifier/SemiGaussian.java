package Classifier;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Random;

import javax.print.Doc;

import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;

import structures.MyPriorityQueue;
import structures._Corpus;
import structures._Doc;
import structures._RankItem;
import structures._SparseFeature;
import utils.Utils;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2D;

public class SemiGaussian extends BaseClassifier{
	protected double[] m_pY;//p(Y), the probabilities of different classes.
	protected double m_labelRatio; // percentage of training data for semi-supervised learning
	protected double m_TLalpha; //Weight coefficient between unlabeled node and labeled node.
	protected double m_TLbeta; //Weight coefficient between unlabeled node and unlabeled node.
	protected double m_M; //Influence of labeled node.
	protected int m_k; // k labeled nodes.
	protected int m_kPrime;//k' unlabeled nodes.
	protected double m_discount = 0.5; // default similarity discount if across different products

	private int m_U, m_L; 
	private double[] m_Y; //The predicted labels for both labeled and unlabeled data.
	private double[] m_Y_U;//The predicted labels for unlabeled data. Init
	private String[][] m_debugOutput;//The two dimension array is used to represent the debug output for all unlabeled data.
	private double[] m_cache; // cache the similarity computation results given the similarity metric is symmetric
	protected MyPriorityQueue<_RankItem> m_kUL, m_kUU; // k nearest neighbors for Unlabeled-Labeled and Unlabeled-Unlabeled
	protected ArrayList<_Doc> m_labeled; // a subset of training set
	
	protected BaseClassifier m_classifier; //Multiple learner.	
	protected PrintWriter m_writer;
	
	/**the parameter used in Zhu's calculating similarity, we want to train it during the training process.***/
	protected double[] m_sigma; 
	protected double[] m_g;
	protected double[] m_diag;
	protected double[][] m_Wij;
	protected double[][] m_g_Wij;
	protected double[][] m_Pij;
	protected double[][] m_PijGradient;
	
	//Default constructor without any default parameters.
	public SemiGaussian(_Corpus c, int classNumber, int featureSize, String classifier){
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
		m_sigma = new double[featureSize];
		m_g = new double[m_sigma.length];
		m_diag = new double[m_sigma.length];
	}	
	
	//Constructor: given k and kPrime
	public SemiGaussian(_Corpus c, int classNumber, int featureSize, String classifier, int k, int kPrime){
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
		m_sigma = new double[featureSize];
		m_g = new double[m_sigma.length];
		m_diag = new double[m_sigma.length];
	}
	
	//Constructor: given k, kPrime, TLalpha and TLbeta
	public SemiGaussian(_Corpus c, int classNumber, int featureSize, String classifier, int k, int kPrime, double TLalhpa, double TLbeta){
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
		m_sigma = new double[featureSize];
		m_g = new double[m_sigma.length];
		m_diag = new double[m_sigma.length];
	}
	
	public SemiGaussian(_Corpus c, int classNumber, int featureSize, String classifier, double ratio, int k, int kPrime, double TLalhpa, double TLbeta){
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
		m_sigma = new double[featureSize];
		m_g = new double[m_sigma.length];
		m_diag = new double[m_sigma.length];
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
//		Arrays.fill(m_pY, 0);
		Arrays.fill(m_sigma, 1);
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
	
	//This train is used to train the sigma in Jerry Zhu's 
	public void train(Collection<_Doc> trainSet){
		m_classifier.train(trainSet);//Multiple learner.
		init();
		
		//Jerry Zhu's wij caculation.
		int[] iflag = {0}, iprint = { -1, 3 };
		double fValue;
		int fSize = m_sigma.length;
		
//		try{
//			do {
//				fValue = calcH(trainSet);
//				LBFGS.lbfgs(fSize, 6, m_sigma, fValue, m_g, false, m_diag, iprint, 1e-4, 1e-20, iflag);
//			} while (iflag[0] != 0);
//		} catch (ExceptionWithIflag e){
//			e.printStackTrace();
//		}
		
		//Randomly pick some training documents as the labeled documents.
		Random r = new Random();
		for (_Doc doc: trainSet){
//			m_pY[doc.getYLabel()]++;
			if(r.nextDouble()<m_labelRatio){
				m_labeled.add(doc);
			}
		}
	}
	
	public void propress(){
		
	}
	
	//Calculate the value of entropy H.
	public double calcH() {
		double fValue = 0;
		for(int i = 0; i < m_U; i++){
			
		}
		return fValue;
	}

	//Calculate the gradient of H with respect to the sigma_d.
	public void calcHGradient() {

	}

	//Calcualte the value of the similarity Wij.
	public double calcWij(_Doc d1, _Doc d2) {
		double Wij = 0, sigmaD = 0;
		_SparseFeature[] spVct1 = d1.getSparse();
		_SparseFeature[] spVct2 = d2.getSparse();
		int pointer1 = 0, pointer2 = 0;
		while (pointer1 < spVct1.length && pointer2 < spVct2.length) {
			_SparseFeature temp1 = spVct1[pointer1];
			_SparseFeature temp2 = spVct2[pointer2];
			if (temp1.getIndex() == temp2.getIndex()) {
				sigmaD = m_sigma[temp1.getIndex()];
				Wij += (temp1.getValue() - temp2.getValue())* (temp1.getValue() - temp2.getValue()) / (sigmaD * sigmaD);
				pointer1++;
			} else if (temp1.getIndex() > temp2.getIndex()) {
				sigmaD = m_sigma[temp2.getIndex()];
				Wij += temp2.getValue() * temp2.getValue() / (sigmaD * sigmaD);
				pointer2++;
			} else {
				sigmaD = m_sigma[temp1.getIndex()];
				Wij += temp1.getValue() * temp1.getValue() / (sigmaD * sigmaD);
				pointer1++;
			}
		}
		Wij = Math.exp(-Wij);
		return Wij;
	}
	
	//Use this to get the gradient of the similarity of Wij.
	public void calcWijGradient() {
		double Wij = 0;
		for (_Doc d1 : m_testSet) {
			for (_Doc d2 : m_testSet) {
				_SparseFeature[] spVct1 = d1.getSparse();
				_SparseFeature[] spVct2 = d2.getSparse();
				int pointer1 = 0, pointer2 = 0;
				while (pointer1 < spVct1.length && pointer2 < spVct2.length) {
					_SparseFeature temp1 = spVct1[pointer1];
					_SparseFeature temp2 = spVct2[pointer2];
					double thetaD = 0;
					if (temp1.getIndex() == temp2.getIndex()) {
						thetaD = m_sigma[temp1.getIndex()];
						m_g[temp1.getIndex()] = 2 * (temp1.getValue() - temp2.getValue())	* (temp1.getValue() - temp2.getValue()) / (thetaD * thetaD);
						pointer1++;
					} else if (temp1.getIndex() > temp2.getIndex()) {
						m_g[temp1.getIndex()] = temp2.getValue()
								* temp2.getValue();
						pointer2++;
					} else {
						Wij += temp1.getValue() * temp1.getValue();
						pointer1++;
					}
				}
			}
		}
	}
	
	
	
	//Test the data set, including the transductive learning process.
	public void test() throws FileNotFoundException{
		double similarity = 0;
		m_L = m_labeled.size();
		m_U = m_testSet.size();
		m_Y = new double[m_U + m_L];
		m_debugOutput = new String[m_U][13];

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
				setCache(j, i, similarity);
			}
			
			for (int j = 0; j < m_L; j++) {
				dj = m_labeled.get(j);
				similarity = Utils.calculateSimilarity(di, dj) * di.getWeight() * dj.getWeight();
				if (!di.sameProduct(m_labeled.get(j)))
					similarity *= m_discount;// differentiate reviews from different products
				setCache(i, m_U + j, similarity);
				setCache(m_U + j, i, similarity);
			}
		}
		SparseDoubleMatrix2D mat = new SparseDoubleMatrix2D(m_U + m_L, m_U	+ m_L);
		/*** Set up structure for k nearest neighbors. ****/
		m_kUU = new MyPriorityQueue<_RankItem>(m_kPrime);
		m_kUL = new MyPriorityQueue<_RankItem>(m_k);
		/**** Construct the C+scale*\Delta matrix and Y vector. ****/
		double scale = -m_TLalpha / (m_k + m_TLbeta * m_kPrime), sum, value;
		for (int i = 0; i < m_U; i++) {
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
			}
			//Put the top 5 unlabeled nearest data into the matrix.
			int topK = 0;
			while(topK < 5){
				_RankItem n = m_kUU.elementAt(topK);
				_Doc d = m_testSet.get(n.m_index);
				m_debugOutput[i][topK]= Integer.toString(d.getYLabel());
				topK++;
			}
			m_kUU.clear();

			// Set the part of labeled and unlabeled nodes. L-U and U-L
			for (int j = 0; j < m_L; j++)
				m_kUL.add(new _RankItem(m_U + j, getCache(i, m_U + j)));
			for (_RankItem n : m_kUL) {
				value = Math.max(n.m_value, mat.getQuick(i, n.m_index) / scale);// recover the original Wij
				mat.setQuick(i, n.m_index, scale * value);
				mat.setQuick(n.m_index, i, scale * value);
				sum += value;
			}
			mat.setQuick(i, i, 1 - scale * sum);
			topK = 0;
			while(topK < 5){
				_RankItem n = m_kUL.elementAt(topK);
				_Doc d = m_trainSet.get((n.m_index-m_U));
				m_debugOutput[i][topK+5]= Integer.toString(d.getYLabel());
				topK++;
			}
			m_kUL.clear();

			// set up the Y vector for unlabeled data
			m_Y[i] = m_classifier.predict(m_testSet.get(i)); // Multiple learner.
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
			m_TPTable[getLabel3(m_Y_U[i])][m_testSet.get(i).getYLabel()] += 1;
			m_debugOutput[i][10] = Integer.toString(m_testSet.get(i).getYLabel());
			m_debugOutput[i][11] = Integer.toString(getLabel3(m_Y_U[i]));
			m_debugOutput[i][12] = m_testSet.get(i).getSource();
		}
		m_precisionsRecalls.add(calculatePreRec(m_TPTable));
		Debugoutput();
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
	public int predict(_Doc doc) { return -1;}//we don't support this
	
	@Override
	public void saveModel(String modelLocation){}//Save the parameters for classification.
}
