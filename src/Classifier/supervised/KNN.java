package Classifier.supervised;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Random;
import structures._Corpus;
import structures._Doc;
import structures._RankItem;
import utils.Utils;
import Classifier.BaseClassifier;

public class KNN extends BaseClassifier{
	int m_k;
	int m_l;
	double[][] m_randomVcts;
	HashMap<Integer, ArrayList<_Doc>> m_buckets;
	
	public KNN(_Corpus c, int classNumber, int featureSize){
		super(c, classNumber, featureSize);
		m_k = 5;
		m_l = 10;
		m_buckets = new HashMap<Integer, ArrayList<_Doc>>();
	}
	
	public KNN(_Corpus c, int classNumber, int featureSize, int k, int l){
		super(c, classNumber, featureSize);
		m_k = k;
		m_l = l;
		m_randomVcts = new double[m_l][featureSize];
		m_buckets = new HashMap<Integer, ArrayList<_Doc>>();
	}

	//Initialize the random vectors.
	protected void init() {
		Random r = new Random();
		double start = -1.0;
		double end = 1.0;
		for(int i = 0; i < m_l; i++){
			for(int j = 0; j < m_featureSize; j++){
				m_randomVcts[i][j] = start + r.nextDouble() * (end - start);
			}
		}
	}
	
	public void setKL(int k, int l){
		m_k = k;
		m_l = l;
		m_randomVcts = new double[m_l][m_featureSize];
	}
	
	@Override
	//Group all the documents based on their hashcodes.
	public void train(Collection<_Doc> trainSet) {
		init();
		for(_Doc d: trainSet){
			int hashCode = getHashCode(d);
			if(m_buckets.containsKey(hashCode)){
				m_buckets.get(hashCode).add(d);
			} else{
				ArrayList<_Doc> docs = new ArrayList<_Doc>();
				docs.add(d);
				m_buckets.put(hashCode, docs);
			}
		}
	}

	//Get the hashcode for every document.
	public int getHashCode(_Doc d){
		int[] hashArray = new int[m_l];
		for(int i = 0; i < m_l; i++){
			double value = Utils.dotProduct(m_randomVcts[i], d.getSparse());
			hashArray[i] = Utils.sgn(value);
		}
		int hashCode = Utils.encode(hashArray);
		return hashCode;
	}
	
	@Override
	public int predict(_Doc doc) {
		int hashCode = getHashCode(doc);
		ArrayList<_Doc> docs = m_buckets.get(hashCode);
		if(docs.size() < m_k){
			System.err.println("L is set too large, tune the parameter.");
			return -1;
		}
		else{
			_RankItem[] similarities = new _RankItem[docs.size()];
			for(int i = 0; i < docs.size(); i++){
				double similarity = Utils.calculateCosineSimilarity(docs.get(i), doc);
				similarities[i] = new _RankItem(i,similarity, docs.get(i).getYLabel());
			}
			Arrays.sort(similarities);
			return findMajority(similarities, m_k);
		}
	}
	
	public String[] predictRandomProjection(_Doc doc) {
		long start = System.currentTimeMillis();
		String[] kNeighbors = new String[m_k];
		int hashCode = getHashCode(doc);
		ArrayList<_Doc> docs = m_buckets.get(hashCode);
		_RankItem[] similarities = new _RankItem[docs.size()];
		for(int i = 0; i < docs.size(); i++){
			double similarity = Utils.calculateCosineSimilarity(docs.get(i), doc);
			similarities[i] = new _RankItem(i,similarity, docs.get(i).getYLabel());
		}
		Arrays.sort(similarities);
		for(int i=0; i < m_k; i++){
			int index = docs.size()-1-i;
			String content = docs.get(similarities[index].m_index).getSource();
			kNeighbors[i] = content;
		}
		long end = System.currentTimeMillis();
		System.out.println("It takes "+(end-start)+" million seconds to find " + m_k + " nearest neighbors for one review.");
		return kNeighbors;
	}
	
	public String[] predictBruteForce(_Doc doc){
		long start = System.currentTimeMillis();
		int size = m_trainSet.size();
		String[] kNeighbors = new String[m_k];
		_RankItem[] neighbors = new _RankItem[size];
		for(int i = 0; i < size; i++){
			double similarity = Utils.calculateCosineSimilarity(m_trainSet.get(i), doc);
			neighbors[i] = new _RankItem(i, similarity);
		}
		Arrays.sort(neighbors);
		for(int i=0; i < m_k; i++){
			int index = size-1-i;
			String content = m_trainSet.get(neighbors[index].m_index).getSource();
			kNeighbors[i] = content;
		}
		long end = System.currentTimeMillis();
		System.out.println("It takes "+(end-start)+" million seconds to find " + m_k + " nearest neighbors for one review.");
		return kNeighbors;
	}

	public int findMajority(_RankItem[] similarities, int k){
		int[] labels = new int[m_k];
		int size = similarities.length;
		for(int i=0; i < m_k; i++){
			labels[i] = similarities[size-1-i].m_label;
		}
		int[] groups = new int[m_classNo];
		for(int i = 0; i < m_k; i++){
			groups[labels[i]]++;
		}
		return Utils.maxOfArrayIndex(groups);
	}

	@Override
	protected void debug(_Doc d) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void saveModel(String modelLocation) {
		// TODO Auto-generated method stub
		
	}

	public void setTrainSet(){
		m_trainSet = m_corpus.getCollection();
	}
	
	public void setTestSet(ArrayList<_Doc> testSet){
		m_testSet = testSet;
	}
}
