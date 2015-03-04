package Classifier;

import liblinear.*;

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
import structures._SparseFeature;
import utils.Utils;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2D;

public class SemiLinear extends BaseClassifier{
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
	private double[] m_Y_U;//The predicted labels for unlabeled data.
	
	private double[] m_cache; // cache the similarity computation results given the similarity metric is symmetric
	protected MyPriorityQueue<_RankItem> m_kUL, m_kUU; // k nearest neighbors for Unlabeled-Labeled and Unlabeled-Unlabeled
	protected ArrayList<_Doc> m_labeled; // a subset of training set
	
	protected BaseClassifier m_classifier; //Multiple learner.	
	protected PrintWriter m_writer;
	
	protected Model m_libModel;
	protected int m_base;
	//Default constructor without any default parameters.
	public SemiLinear(_Corpus c, int classNumber, int featureSize, String classifier){
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
		m_base = featureSize;
	}	
	
	public SemiLinear(_Corpus c, int classNumber, int featureSize, String classifier, double ratio, int k, int kPrime, double TLalhpa, double TLbeta){
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
		m_base = featureSize;
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
		//Train the m_LinearWeight first with libliear.
		m_libModel = trainLibLinear(3);
		
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
	
	//In this training process, we want to get the weight of all pairs of samples.
	public Model trainLibLinear(int bound){
		//In the problem, the size of feature size is m*m.
		double sampleSize = m_trainSet.size()*(m_trainSet.size()-1) / 2;
		Feature[][] featureMatrix = new Feature[(int) sampleSize][];
		int matrixIndex = 0;
		double[] target = new double[(int) sampleSize];
		for(int i = 0; i < m_trainSet.size(); i++){
			_Doc d1 = m_trainSet.get(i);
			for(int j = i+1; j < m_trainSet.size(); j++){
				_Doc d2 = m_trainSet.get(j);
				featureMatrix[matrixIndex] = calculateLibFeature(d1, d2);
				if(d1.getYLabel() == d2.getYLabel()){
					target[matrixIndex] = 3; //If similiar, 1 + 2 = 3
				} else if(Math.abs(d1.getYLabel() - d2.getYLabel())>bound){
					target[matrixIndex] = 1; //If dissimilar, -1 + 2 = 1
				} else target[matrixIndex] = 2; //Else, 1 + 2 = 2
				matrixIndex++;
			}
		}
		double C = 1.0, eps = 0.01;
		Parameter libParameter = new Parameter(SolverType.L2R_LR, C, eps);
		
		Problem libProblem = new Problem();
		libProblem.l = (int) sampleSize;
		libProblem.n = m_featureSize * m_featureSize;
		libProblem.x = featureMatrix;
		libProblem.y = target;
		Model model = Linear.train(libProblem, libParameter);
		return model;
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
	
	//Calculate the new sample according to two documents.
	public Feature[] calculateLibFeature(_Doc d1, _Doc d2){
		_SparseFeature[] spVct1 = d1.getSparse();
		_SparseFeature[] spVct2 = d2.getSparse();
		Feature[] features = new Feature[spVct1.length * spVct2.length];
		int index = 0;
		double value = 0;
		for(int i = 0; i < spVct1.length; i++){
			for(int j = 0; j < spVct2.length; j++){
				index = (int) (Math.pow(m_base, spVct1[i].getIndex()) + spVct2[j].getIndex());
				value = spVct1[i].getValue() * spVct2[j].getValue();
				features[i * spVct2.length + j] = new FeatureNode(index, value);
			}
		}
		return features;
	}
	
	//Test the data set, including the transductive learning process.
	public void test() throws FileNotFoundException{
		double similarity = 0;
		m_L = m_labeled.size();
		m_U = m_testSet.size();
		m_Y = new double[m_U + m_L];
		m_Y_U = new double[m_U];
		
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
				//Instead of using cosine similarity, we use the new trained classifer to get the M distance.
				//similarity = Utils.calculateSimilarity(di, dj) * di.getWeight() * dj.getWeight();
				similarity = Linear.predict(m_libModel, calculateLibFeature(di, dj));
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
		}
		m_precisionsRecalls.add(calculatePreRec(m_TPTable));
	}
	
	/**Different getLabel methods.**/
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
