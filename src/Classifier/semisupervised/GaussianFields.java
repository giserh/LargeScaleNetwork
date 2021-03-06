package Classifier.semisupervised;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Random;

import structures.MyPriorityQueue;
import structures._Corpus;
import structures._Doc;
import structures._RankItem;
import structures._SparseFeature;
import utils.Utils;
import Classifier.BaseClassifier;
import Classifier.supervised.LogisticRegression;
import Classifier.supervised.NaiveBayes;
import Classifier.supervised.PRLogisticRegression;
import Classifier.supervised.SVM;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2D;

public class GaussianFields extends BaseClassifier {
	
	double m_alpha; //Weight coefficient between unlabeled node and labeled node.
	double m_beta; //Weight coefficient between unlabeled node and unlabeled node.
	double m_M; //Influence of labeled node.
	int m_k; // k labeled nodes.
	int m_kPrime;//k' unlabeled nodes.
	
	int m_U, m_L;
	double[] m_cache; // cache the similarity computation results given the similarity metric is symmetric
	double[] m_fu;
	double[] m_Y; // true label for the labeled data and pseudo label from base learner
	SparseDoubleMatrix2D m_graph;
	
	MyPriorityQueue<_RankItem> m_kUL, m_kUU; // k nearest neighbors for Unlabeled-Labeled and Unlabeled-Unlabeled
	ArrayList<_Doc> m_labeled; // a subset of training set
	protected double m_labelRatio; // percentage of training data for semi-supervised learning
	
	BaseClassifier m_classifier; //Multiple learner.
	double[] m_pY;//p(Y), the probabilities of different classes.
	double[] m_pYSum; //\sum_i exp(-|c-fu(i)|)
	
	double m_discount = 0; // default similarity discount if across different products
	double[][] m_A; //The matrix used to store the result of metric learning.
	int m_POSTagging; //If postagging is non-zero and projected features are used to do similarity calculation.
	ArrayList<int[][]> m_reviewsStat; //For one review, the number of reviews of the same producte ID, the number of reviews of similarity > 0.
	HashMap<Integer, String> m_IndexFeature;//For debug purpose.
	
	//Randomly pick 10% of all the training documents.
	public GaussianFields(_Corpus c, int classNumber, int featureSize, String classifier){
		super(c, classNumber, featureSize);
		
		m_labelRatio = 0.2;
		m_alpha = 1.0;
		m_beta = 0.1;
		m_M = 10000;
		m_k = 100;
		m_kPrime = 50;	
		m_labeled = new ArrayList<_Doc>();
		m_pY = new double[classNumber];
		m_pYSum = new double[classNumber];
		m_POSTagging = 0;
		setClassifier(classifier);
		m_reviewsStat = new ArrayList<int[][]>();
	}	
	
	public GaussianFields(_Corpus c, int classNumber, int featureSize, String classifier, double ratio, int k, int kPrime){
		super(c, classNumber, featureSize);
		
		m_labelRatio = ratio;
		m_alpha = 1.0;
		m_beta = 0.1;
		m_M = 10000;
		m_k = k;
		m_kPrime = kPrime;	
		m_labeled = new ArrayList<_Doc>();
		m_pY = new double[classNumber];
		m_pYSum = new double[classNumber];
		m_POSTagging = 0;
		setClassifier(classifier);
		m_reviewsStat = new ArrayList<int[][]>();

	}
	
	public void setPOSTagging(int a){
		m_POSTagging = a;
	}
	
	public void setDebugPrinters(String wrongRW, String wrongSVM, String FuSVM) throws FileNotFoundException, UnsupportedEncodingException {
		m_writerWrongRW = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(wrongRW, false), "UTF-8"));
		m_writerWrongSVM = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(wrongSVM, false), "UTF-8"));
		m_writerFuSVM = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(FuSVM, false), "UTF-8"));
	}
	
	@Override
	public String toString() {
		return String.format("Gaussian Fields with matrix inversion [C:%s, kUL:%d, kUU:%d, r:%.3f, alpha:%.3f, beta:%.3f]", m_classifier, m_k, m_kPrime, m_labelRatio, m_alpha, m_beta);
	}
	
	private void setClassifier(String classifier) {
		if (classifier.equals("NB"))
			m_classifier = new NaiveBayes(null, m_classNo, m_featureSize);
		else if (classifier.equals("LR"))
			m_classifier = new LogisticRegression(null, m_classNo, m_featureSize);
		else if (classifier.equals("PR-LR"))
			m_classifier = new PRLogisticRegression(null, m_classNo, m_featureSize, 1.0);
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
		Arrays.fill(m_pYSum, 0);
	}

	public void setMatrixA(double[][] A){
		this.m_A = A;
	}
	
	//Train the data set.
	public void train(Collection<_Doc> trainSet){
		init();
		
		m_classifier.train(trainSet);
		
		//Randomly pick some training documents as the labeled documents.
		Random r = new Random();
		for (_Doc doc:trainSet){
			m_pY[doc.getYLabel()]++;
			if(r.nextDouble()<m_labelRatio)
				m_labeled.add(doc);
		}
		
		//estimate the prior of p(y=c)
		Utils.scaleArray(m_pY, 1.0/Utils.sumOfArray(m_pY));
	}
	
	void initCache() {
		int size = m_U*(2*m_L+m_U-1)/2;//specialized for the current matrix structure
		if (m_cache==null || m_cache.length<size)
			m_cache = new double[m_U*(2*m_L+m_U-1)/2]; // otherwise we can reuse the current memory space
	}
	
	int encode(int i, int j) {
		if (i>j) {//swap
			int t = i;
			i = j;
			j = t;
		}
		int sum = 0;
		if(i == 0) 
			sum = j - 1;
		else{
			for(int m = 0; m < i; m++){
				sum += m_U + m_L - m - 1;
			}
			sum += j - i - 1;
		}
		return sum;//specialized for the current matrix structure
	}
	
	public void debugEncoder() {
		m_U = 10;
		m_L = 5;
		
		for(int i=0; i<m_U; i++) {
			for(int j=i+1; j<m_U; j++) {
				System.out.print(encode(i,j) + " ");
			}
			
			for(int j=0; j<m_L; j++) {
				System.out.print(encode(i,m_U+j) + " ");
			}
			
			System.out.println();
		}
	}
	
	void setCache(int i, int j, double v) {
		m_cache[encode(i,j)] = v;
	}
	
	double getCache(int i, int j) {
		return m_cache[encode(i,j)];
	}
	
	protected double getSimilarity(_Doc di, _Doc dj) {
		if(m_A != null)
			return Utils.calculateMetricLearning(di, dj, m_A);
		else if(m_POSTagging == 0)
			return Utils.calculateSimilarity(di, dj);
		else 
			return Utils.calculateProjSimilarity(di, dj);
//		return Math.random();//just for debugging purpose
		
	}
	
	protected void constructGraph(boolean createSparseGraph) {
		double similarity = 0;
		m_L = m_labeled.size();
		m_U = m_testSet.size();
		int[][] oneReviewsStat = new int[m_U][4];
		
		/*** Set up cache structure for efficient computation. ****/
		initCache();
		if (m_fu==null || m_fu.length<m_U)
//			m_fu = new double[m_U]; //otherwise we can reuse the current memory
			m_fu = new double[m_U];
		if (m_Y==null || m_Y.length<m_U+m_L)
			m_Y = new double[m_U+m_L];
		
		/*** pre-compute the full similarity matrix (except the diagonal). ****/
		_Doc di, dj;
		for (int i = 0; i < m_U; i++) {
			di = m_testSet.get(i);
			for (int j = i + 1; j < m_U; j++) {// to save computation since our similarity metric is symmetric
				dj = m_testSet.get(j);
				similarity = getSimilarity(di, dj) * di.getWeight() * dj.getWeight();
			
//				if (!di.sameProduct(dj))
//					similarity *= m_discount;// differentiate reviews from different products have similarities 0.
//				else {
//					oneReviewsStat[i][0]++; // If there are same product, then add one for neighbors under one product.
//					oneReviewsStat[j][0]++; // If there are same product, then add one for neighbors under one product.
//				}
//				if (similarity != 0){
//					oneReviewsStat[i][1]++;
//					oneReviewsStat[j][1]++;
//				}
				setCache(i, j, similarity);
			}

			for (int j = 0; j < m_L; j++) {
				dj = m_labeled.get(j);
				similarity = getSimilarity(di, dj) * di.getWeight() * dj.getWeight();
				
//				if (!di.sameProduct(m_labeled.get(j)))
//					similarity *= m_discount;// differentiate reviews from different products have similarities 0.
//				else {
//					oneReviewsStat[i][2]++; // If there are same product, then add one for neighbors under one product.
//				}
//				if (similarity != 0){
//					oneReviewsStat[i][3]++;
//				}
				setCache(i, m_U + j, similarity);
			}
			
			//set up the Y vector for unlabeled data
//			m_Y[i] = 1; //Multiple learner.//
			m_Y[i] = m_classifier.predict(m_testSet.get(i));
		}	
		m_reviewsStat.add(oneReviewsStat);
		//set up the Y vector for labeled data
		for(int i=m_U; i<m_L+m_U; i++)
			m_Y[i] = m_labeled.get(i-m_U).getYLabel();
		
		/***Set up structure for k nearest neighbors.****/
		m_kUU = new MyPriorityQueue<_RankItem>(m_kPrime);
		m_kUL = new MyPriorityQueue<_RankItem>(m_k);
		
		/***Set up document mapping for debugging purpose***/
		if (m_debugOutput!=null) {
			for (int i = 0; i < m_U; i++) 
				m_testSet.get(i).setID(i);//record the current position
		}
		
		if (!createSparseGraph) {
			System.out.println("Nearest neighbor graph construction finished!");
			return;//stop here if we want to save memory and construct the graph on the fly (space speed trade-off)
		}
		
		m_graph = new SparseDoubleMatrix2D(m_U+m_L, m_U+m_L);//we have to create this every time with exact dimension
		
		/****Construct the C+scale*\Delta matrix and Y vector.****/
		double scale = -m_alpha / (m_k + m_beta*m_kPrime), sum, value;
		for(int i = 0; i < m_U; i++) {
			//set the part of unlabeled nodes. U-U
			for(int j=0; j<m_U; j++) {
				if (j==i)
					continue;
				m_kUU.add(new _RankItem(j, getCache(i,j)));
			}
			
			sum = 0;
			for(_RankItem n:m_kUU) {
				value = Math.max(m_beta*n.m_value, m_graph.getQuick(i, n.m_index)/scale);//recover the original Wij
				m_graph.setQuick(i, n.m_index, scale * value);
				m_graph.setQuick(n.m_index, i, scale * value);
				sum += value;
			}
			m_kUU.clear();
			
			//Set the part of labeled and unlabeled nodes. L-U and U-L
			for(int j=0; j<m_L; j++) 
				m_kUL.add(new _RankItem(m_U+j, getCache(i,m_U+j)));
			
			for(_RankItem n:m_kUL) {
				value = Math.max(n.m_value, m_graph.getQuick(i, n.m_index)/scale);//recover the original Wij
				m_graph.setQuick(i, n.m_index, scale * value);
				m_graph.setQuick(n.m_index, i, scale * value);
				sum += value;
			}
			m_graph.setQuick(i, i, 1-scale*sum);
			m_kUL.clear();
		}
		
		for(int i=m_U; i<m_L+m_U; i++) {
			sum = 0;
			for(int j=0; j<m_U; j++) 
				sum += m_graph.getQuick(i, j);
			m_graph.setQuick(i, i, m_M-sum); // scale has been already applied in each cell
		}
		
		System.out.println("Nearest neighbor graph construction finished!");
	}
	
	//Test the data set.
	@Override
	public double test(){	
		/***Construct the nearest neighbor graph****/
		constructGraph(true);
		
		/***Perform matrix inverse.****/
		DenseDoubleAlgebra alg = new DenseDoubleAlgebra();
		DoubleMatrix2D result = alg.inverse(m_graph);
		
		/***setting up the corresponding weight for the true labels***/
		for(int i=m_U; i<m_L+m_U; i++)
			m_Y[i] *= m_M;
		
		/***get some statistics***/
		for(int i = 0; i < m_U; i++){
			double pred = 0;
			for(int j=0; j<m_U+m_L; j++)
				pred += result.getQuick(i, j) * m_Y[j];			
			m_fu[i] = pred;//prediction for the unlabeled based on the labeled data and pseudo labels
			
			for(int j=0; j<m_classNo; j++)
				m_pYSum[j] += Math.exp(-Math.abs(j-m_fu[i]));			
		}
		
		/***evaluate the performance***/
		double acc = 0;
		int pred, ans;
		for(int i = 0; i < m_U; i++) {
			pred = getLabel(m_fu[i]);
			ans = m_testSet.get(i).getYLabel();
			m_TPTable[pred][ans] += 1;
			
			if (pred != ans) {
				if (m_debugOutput!=null)
					debug(m_testSet.get(i));
			} else 
				acc ++;
		}
		m_precisionsRecalls.add(calculatePreRec(m_TPTable));
		
		return acc/m_U;
	}
	
	/**Different getLabel methods.**/
	//This is the original getLabel: -|c-p(c)|
	int getLabel(double pred) {
		for(int i=0; i<m_classNo; i++)
			m_cProbs[i] = -Math.abs(i-pred); //-|c-p(c)|
		return Utils.maxOfArrayIndex(m_cProbs);
	}
	
	//p(c) * exp(-|c-f(u_i)|)/sum_j{exp(-|c-f(u_j))} j represents all unlabeled data
	int getLabel3(double pred){
		for(int i = 0; i < m_classNo; i++)			
			m_cProbs[i] = m_pY[i] * Math.exp(-Math.abs(i-pred)) / m_pYSum[i];
		return Utils.maxOfArrayIndex(m_cProbs);
	}
	
	//exp(-|c-f(u_i)|)/sum_j{exp(-|c-f(u_j))} j represents all unlabeled data, without class probabilities.
	int getLabel4(double pred) {		
		for (int i = 0; i < m_classNo; i++)
			m_cProbs[i] = Math.exp(-Math.abs(i - pred)) / m_pYSum[i];
		return Utils.maxOfArrayIndex(m_cProbs);
	}
	
	//Construct the look-up table for the later debugging use.
	public void setFeaturesLookup(HashMap<String, Integer> featureNameIndex){
		m_IndexFeature = new HashMap<Integer, String>();
		for(String f: featureNameIndex.keySet()){
			m_IndexFeature.put(featureNameIndex.get(f), f);
		}
	}
	@Override
	protected void debug(_Doc d){
		int id = d.getID();
		_SparseFeature[] dsfs = d.getSparse();
		_RankItem item;
		_Doc neighbor;
		double sim, wijSumU=0, wijSumL=0;
		
		try {
			m_debugWriter.write("============================================================================\n");
			m_debugWriter.write(String.format("Label:%d, fu:%.4f, getLabel1:%d, getLabel3:%d, SVM:%d, Content:%s\n", d.getYLabel(), m_fu[id], getLabel(m_fu[id]), getLabel3(m_fu[id]), (int)m_Y[id], d.getSource()));
			
			for(int i = 0; i< dsfs.length; i++){
				String feature = m_IndexFeature.get(dsfs[i].getIndex());
				m_debugWriter.write(String.format("(%s %.4f),", feature, dsfs[i].getValue()));
			}
			m_debugWriter.write("\n");
			
			//find top five labeled
			/****Construct the top k labeled data for the current data.****/
			for (int j = 0; j < m_L; j++){
				m_kUL.add(new _RankItem(j, getCache(id, m_U + j)));
			}
	
			/****Get the sum of kUL******/
			for(_RankItem n: m_kUL)
				wijSumL += n.m_value; //get the similarity between two nodes.
			
			/****Get the top 5 elements from kUL******/
			m_debugWriter.write("*************************Labeled data*************************************\n");
			for(int k=0; k < 5; k++){
				item = m_kUL.get(k);
				neighbor = m_labeled.get(item.m_index);
				sim = item.m_value/wijSumL;
				
				//Print out the sparse vectors of the neighbors.
				m_debugWriter.write(String.format("Label:%d, Similarity:%.4f\n", neighbor.getYLabel(), sim));
				m_debugWriter.write(neighbor.getSource()+"\n");
				_SparseFeature[] sfs = neighbor.getSparse();
				int pointer1 = 0, pointer2 = 0;
				//Find out all the overlapping features and print them out.
				while(pointer1 < dsfs.length && pointer2 < sfs.length){
					_SparseFeature tmp1 = dsfs[pointer1];
					_SparseFeature tmp2 = sfs[pointer2];
					if(tmp1.getIndex() == tmp2.getIndex()){
						String feature = m_IndexFeature.get(tmp1.getIndex());
						m_debugWriter.write(String.format("(%s %.4f),", feature, tmp2.getValue()));
						pointer1++;
						pointer2++;
					} else if(tmp1.getIndex() < tmp2.getIndex())
						pointer1++;
					else pointer2++;
				}
				m_debugWriter.write("\n");
			}
			m_kUL.clear();
			
			//find top five unlabeled
			/****Construct the top k' unlabeled data for the current data.****/
			for (int j = 0; j < m_U; j++) {
				if (j == id)
					continue;
				m_kUU.add(new _RankItem(j, getCache(id, j)));
			}
			
			/****Get the sum of k'UU******/
			for(_RankItem n: m_kUU)
				wijSumU += n.m_value; //get the similarity between two nodes.
			
			/****Get the top 5 elements from k'UU******/
			m_debugWriter.write("*************************Unlabeled data*************************************\n");
			for(int k=0; k<5; k++){
				item = m_kUU.get(k);
				neighbor = m_testSet.get(item.m_index);
				sim = item.m_value/wijSumU;
				
				m_debugWriter.write(String.format("True Label:%d, f_u:%.4f, Similarity:%.4f\n", neighbor.getYLabel(), m_fu[neighbor.getID()], sim));
				m_debugWriter.write(neighbor.getSource()+"\n");
				_SparseFeature[] sfs = neighbor.getSparse();
				int pointer1 = 0, pointer2 = 0;
				//Find out all the overlapping features and print them out.
				while(pointer1 < dsfs.length && pointer2 < sfs.length){
					_SparseFeature tmp1 = dsfs[pointer1];
					_SparseFeature tmp2 = sfs[pointer2];
					if(tmp1.getIndex() == tmp2.getIndex()){
						String feature = m_IndexFeature.get(tmp1.getIndex());
						m_debugWriter.write(String.format("(%s %.4f),", feature, tmp2.getValue()));
						pointer1++;
						pointer2++;
					} else if(tmp1.getIndex() < tmp2.getIndex())
						pointer1++;
					else pointer2++;
				}
				m_debugWriter.write("\n");
			}
			m_kUU.clear();
		} catch (IOException e) {
			e.printStackTrace();
		}
	} 

	@Override
	public int predict(_Doc doc) {
		return -1; //we don't support this in transductive learning
	}
	
	//Save the parameters for classification.
	@Override
	public void saveModel(String modelLocation){
		
	}
	
//	public static void main(String[] args) {
//		GaussianFields test = new GaussianFields(null, 1, 1, "NB");
//		test.debugEncoder();
//	}
	
	//Print out the stat data of reviews of the same product.
	public void printReviewStat(String filename) throws FileNotFoundException{
		if (filename==null || filename.isEmpty())
			return;
		
		PrintWriter writer = new PrintWriter(new File(filename));
		if(m_reviewsStat.size() != 10){
			System.out.println("Error!!");
			return;
		}
		
		for(int i = 0; i < m_reviewsStat.size(); i++){
			int[][] tmp = m_reviewsStat.get(i);
			writer.write("unlabeled neighbors, unlabled non-zero sim, labeled neigbors, labeled non-zero\n");
			for(int j = 0; j < tmp.length; j++)
				writer.write(String.format("%d, %d, %d, %d\n", tmp[j][0], tmp[j][1], tmp[j][2], tmp[j][3]));
		}
		writer.close();
	}
}
