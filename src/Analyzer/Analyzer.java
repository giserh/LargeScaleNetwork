package Analyzer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;


import structures._Corpus;
import structures._Doc;
import structures._SparseFeature;
import structures._stat;
import utils.Utils;

public abstract class Analyzer {
	
	protected _Corpus m_corpus;
	protected int m_classNo; //This variable is just used to init stat for every feature. How to generalize it?
	int[] m_classMemberNo; //Store the number of members in a class.
	protected int m_Ngram; 
	
	protected ArrayList<String> m_featureNames; //ArrayList for features
	protected HashMap<String, Integer> m_featureNameIndex;//key: content of the feature; value: the index of the feature
	protected HashMap<String, _stat> m_featureStat; //Key: feature Name; value: the stat of the feature
	/* Indicate if we can allow new features.After loading the CV file, the flag is set to true, 
	 * which means no new features will be allowed.*/
	protected boolean m_isCVLoaded;
	
	//minimal length of indexed document
	protected int m_lengthThreshold;
	
	/** for time-series features **/
	//The length of the window which means how many labels will be taken into consideration.
	private LinkedList<_Doc> m_preDocs;	
	
	private ArrayList<Double> m_similar;
	private ArrayList<Double> m_dissimilar;
	
	public Analyzer(int classNo, int minDocLength) {
		m_corpus = new _Corpus();
		
		m_classNo = classNo;
		m_classMemberNo = new int[classNo];
		
		m_featureNames = new ArrayList<String>();
		m_featureNameIndex = new HashMap<String, Integer>();//key: content of the feature; value: the index of the feature
		m_featureStat = new HashMap<String, _stat>();
		
		m_lengthThreshold = minDocLength;
		
		m_preDocs = new LinkedList<_Doc>();
		m_dissimilar = new ArrayList<Double>();
		m_similar = new ArrayList<Double>();
	}	
	
	public void reset() {
		Arrays.fill(m_classMemberNo, 0);
		m_featureNames.clear();
		m_featureNameIndex.clear();
		m_featureStat.clear();
		m_corpus.reset();
	}
	
	//Load the features from a file and store them in the m_featurNames.@added by Lin.
	protected boolean LoadCV(String filename) {
		System.out.println("--------------------------------------------------------------------------------------");
		if (filename==null || filename.isEmpty())
			return false;
		
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;
			if((line = reader.readLine()) == null) 
				System.err.println("Empty line detected!!");
			while ((line = reader.readLine()) != null) {
				if (line.startsWith("#")){
					if (line.startsWith("#NGram")) {//has to be decoded
						int pos = line.indexOf(':');
						m_Ngram = Integer.valueOf(line.substring(pos+1));
					}
						
				} else
					expandVocabulary(line);
			}
			reader.close();
			
			System.out.format("%d feature words loaded from %s...\n", m_featureNames.size(), filename);
			m_isCVLoaded = true;
			
			return true;
		} catch (IOException e) {
			System.err.format("[Error]Failed to open file %s!!", filename);
			return false;
		}
	}
	
	//Load the matrix from the matlab result.
	public double[][] loadMatrixA(String filename){
		int featureSize = m_corpus.getFeatureSize();
		double[][] A = new double[featureSize][featureSize];
		int count = 0;
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;
			while ((line = reader.readLine()) != null) {
				double[] tmpD = new double[featureSize];
				String[] tmpS = line.split("\\s+");
				if(featureSize == tmpS.length){
					for(int i = 0; i < featureSize; i++){
						tmpD[i] = Double.parseDouble(tmpS[i]);
					}
				A[count] = tmpD;
				count++;
				} else
					System.err.println("Matrix A: size(A[i]) does not match with feature size!");
			}
			if(count != featureSize)
				System.err.println("Matrix A: size(A) does not match with feature size!");
			reader.close();
			System.out.format("Matrix A is loaded from %s successfully!!", filename);
		} catch (IOException e) {
			System.err.format("[Error]Failed to open file %s!!", filename);
		}
		return A;
	}
	//Load all the files in the directory.
	public void LoadDirectory(String folder, String suffix) throws IOException {
		File dir = new File(folder);
		for (File f : dir.listFiles()) {
			if (f.isFile() && f.getName().endsWith(suffix)) {
				LoadDoc(f.getAbsolutePath());
			} else if (f.isDirectory())
				LoadDirectory(f.getAbsolutePath(), suffix);
		}
		System.out.println();
	}
	
	abstract public void LoadDoc(String filename);
	
	//Save all the features and feature stat into a file.
	protected void SaveCVStat(String finalLocation) throws FileNotFoundException{
		if (finalLocation==null || finalLocation.isEmpty())
			return;
		
		PrintWriter writer = new PrintWriter(new File(finalLocation));
		for(int i = 0; i < m_featureNames.size(); i++){
			writer.print(m_featureNames.get(i));
			_stat temp = m_featureStat.get(m_featureNames.get(i));
			for(int j = 0; j < temp.getDF().length; j++)
				writer.print("\t" + temp.getDF()[j]);
			for(int j = 0; j < temp.getTTF().length; j++)
				writer.print("\t" + temp.getTTF()[j]);
			writer.println();
		}
		writer.close();
	}
	
	//Add one more token to the current vocabulary.
	protected void expandVocabulary(String token) {
		m_featureNameIndex.put(token, m_featureNames.size()); // set the index of the new feature.
		m_featureNames.add(token); // Add the new feature.
		m_featureStat.put(token, new _stat(m_classNo));
	}
		
	//Return corpus without parameter and feature selection.
	public _Corpus returnCorpus(String finalLocation) throws FileNotFoundException {
		SaveCVStat(finalLocation);
		
		for(int c:m_classMemberNo)
			System.out.print(c + " ");
		System.out.println();
		
		return getCorpus();
	}
	
	public _Corpus getCorpus() {
		//store the feature names into corpus
		m_corpus.setFeatures(m_featureNames);
		m_corpus.setMasks(); // After collecting all the documents, shuffle all the documents' labels.
		return m_corpus;
	}
	
	//Give the option, which would be used as the method to calculate feature value and returned corpus, calculate the feature values.
	public void setFeatureValues(String fValue, int norm) {
		ArrayList<_Doc> docs = m_corpus.getCollection(); // Get the collection of all the documents.
		int N = docs.size();
		if (fValue.equals("TF")){
			//the original feature is raw TF
		} else if (fValue.equals("TFIDF")) {
			for (int i = 0; i < docs.size(); i++) {
				_Doc temp = docs.get(i);
				_SparseFeature[] sfs = temp.getSparse();
				for (_SparseFeature sf : sfs) {
					String featureName = m_featureNames.get(sf.getIndex());
					_stat stat = m_featureStat.get(featureName);
					double TF = sf.getValue() / temp.getTotalDocLength();// normalized TF
					double DF = Utils.sumOfArray(stat.getDF());
					double TFIDF = TF * Math.log((N + 1) / DF);
					sf.setValue(TFIDF);
				}
			}
		} else if (fValue.equals("BM25")) {
			double k1 = 1.5; // [1.2, 2]
			double b = 0.75; // (0, 1000]
			// Iterate all the documents to get the average document length.
			double navg = 0;
			for (int k = 0; k < N; k++)
				navg += docs.get(k).getTotalDocLength();
			navg /= N;

			for (int i = 0; i < docs.size(); i++) {
				_Doc temp = docs.get(i);
				_SparseFeature[] sfs = temp.getSparse();
				double n = temp.getTotalDocLength() / navg;
				for (_SparseFeature sf : sfs) {
					String featureName = m_featureNames.get(sf.getIndex());
					_stat stat = m_featureStat.get(featureName);
					double TF = sf.getValue();
					double DF = Utils.sumOfArray(stat.getDF());//
					double BM25 = Math.log((N - DF + 0.5) / (DF + 0.5)) * TF * (k1 + 1) / (k1 * (1 - b + b * n) + TF);
					if (Double.isNaN(BM25))
						System.out.println("Nan detected!!");
					sf.setValue(BM25);
				}
			}
		} else if (fValue.equals("PLN")) {
			double s = 0.5; // [0, 1]
			// Iterate all the documents to get the average document length.
			double navg = 0;
			for (int k = 0; k < N; k++)
				navg += docs.get(k).getTotalDocLength();
			navg /= N;

			for (int i = 0; i < docs.size(); i++) {
				_Doc temp = docs.get(i);
				_SparseFeature[] sfs = temp.getSparse();
				double n = temp.getTotalDocLength() / navg;
				for (_SparseFeature sf : sfs) {
					String featureName = m_featureNames.get(sf.getIndex());
					_stat stat = m_featureStat.get(featureName);
					double TF = sf.getValue();
					double DF = Utils.sumOfArray(stat.getDF());
					double PLN = (1 + Math.log(1 + Math.log(TF)) / (1 - s + s * n)) * Math.log((N + 1) / DF);
					sf.setValue(PLN);
				}
			}
		} else {
			//The default value is just keeping the raw count of every feature.
			System.out.println("No feature value is set, keep the raw count of every feature.");
		}
		
		//rank the documents by product and time in all the cases
		Collections.sort(m_corpus.getCollection());
		if (norm == 1){
			for(_Doc d:docs)			
				Utils.L1Normalization(d.getSparse());
		} else if(norm == 2){
			for(_Doc d:docs)			
				Utils.L2Normalization(d.getSparse());
		} else {
			System.out.println("No normalizaiton is adopted here or wrong parameters!!");
		}
		
		System.out.format("Text feature generated for %d documents...\n", m_corpus.getSize());
	}
	
	//Select the features and store them in a file.
	public void featureSelection(String location, String featureSelection, double startProb, double endProb, int threshold) throws FileNotFoundException {
		FeatureSelector selector = new FeatureSelector(startProb, endProb, threshold);

		System.out.println("--------------------------------------------------------------------------------------");
		if (featureSelection.equals("DF"))
			selector.DF(m_featureStat);
		else if (featureSelection.equals("IG"))
			selector.IG(m_featureStat, m_classMemberNo);
		else if (featureSelection.equals("MI"))
			selector.MI(m_featureStat, m_classMemberNo);
		else if (featureSelection.equals("CHI"))
			selector.CHI(m_featureStat, m_classMemberNo);
		
		m_featureNames = selector.getSelectedFeatures();
		SaveCV(location, featureSelection, startProb, endProb, threshold); // Save all the features and probabilities we get after analyzing.
		System.out.println(m_featureNames.size() + " features are selected!");
		
		//clear memory for next step feature construction
		reset();
		LoadCV(location);//load the selected features
	}
	
	//Save all the features into a file.
	protected void SaveCV(String featureLocation, String featureSelection, double startProb, double endProb, int threshold) throws FileNotFoundException {
		if (featureLocation==null || featureLocation.isEmpty())
			return;
		
		System.out.format("Saving controlled vocabulary to %s...\n", featureLocation);
		PrintWriter writer = new PrintWriter(new File(featureLocation));
		//print out the configurations as comments
		writer.format("#NGram:%d\n", m_Ngram);
		writer.format("#Selection:%s\n", featureSelection);
		writer.format("#Start:%f\n", startProb);
		writer.format("#End:%f\n", endProb);
		writer.format("#DF_Cut:%d\n", threshold);
		
		//print out the features
		for (int i = 0; i < m_featureNames.size(); i++)
			writer.println(m_featureNames.get(i));
		writer.close();
	}
	
	//Return the number of features.
	public int getFeatureSize(){
		return m_featureNames.size();
	}
	
	//Sort the documents.
	public void setTimeFeatures(int window){//must be called before return corpus
		if (window<1) 
			return;
		
		//Sort the documents according to time stamps.
		ArrayList<_Doc> docs = m_corpus.getCollection();
		
		/************************time series analysis***************************/
		double norm = 1.0 / m_classMemberNo.length, avg = 0;
		int count = 0;//for computing the moving average
		String lastItemID = null;
		for(int i = 0; i < docs.size(); i++){
			_Doc doc = docs.get(i);			
			
			if (lastItemID == null)
				lastItemID = doc.getItemID();
			else if (lastItemID != doc.getItemID()) {
				m_preDocs.clear(); // reviews for a new category of products
				lastItemID = doc.getItemID();
				
				//clear for moving average
				avg = 0;
				count = 0;
			}
			
			avg += doc.getYLabel();
			count += 1;
			
			if(m_preDocs.size() < window){
				m_preDocs.add(doc);
				m_corpus.removeDoc(i);
				m_classMemberNo[doc.getYLabel()]--;
				i--;
			} else{
				doc.createSpVctWithTime(m_preDocs, m_featureNames.size(), avg/count, norm);
				m_preDocs.remove();
				m_preDocs.add(doc);
			}
		}
		System.out.format("Time-series feature set for %d documents!\n", m_corpus.getSize());
	}
	
	// added by Md. Mustafizur Rahman for Topic Modelling
	public double[] getBackgroundProb()
	{
		double back_ground_probabilty [] = new double [m_featureNameIndex.size()];
		
		for(int i = 0; i<m_featureNameIndex.size();i++)
		{
			String featureName = m_featureNames.get(i);
			_stat stat =  m_featureStat.get(featureName);
			back_ground_probabilty[i] = Utils.sumOfArray(stat.getTTF());
		}
		
		double sum = Utils.sumOfArray(back_ground_probabilty) + back_ground_probabilty.length;//add one smoothing
		for(int i = 0; i<m_featureNameIndex.size();i++)
			back_ground_probabilty[i] = (1.0 + back_ground_probabilty[i]) / sum;
		return back_ground_probabilty;
	}
	
	//Print the sparse for matlab to generate A.
	public void printXY(String xFile, String yFile) throws FileNotFoundException{
		PrintWriter writer1 = new PrintWriter(new File(xFile));
		PrintWriter writer2 = new PrintWriter(new File(yFile));
		int count = 1;
		for(_Doc d: m_corpus.getCollection()){
			for(_SparseFeature sf: d.getSparse()){
				writer1.write(count + "," + (sf.getIndex() + 1) + "," + sf.getValue() + "\n");
			}
			writer2.write(count + "," + 1 + "," + d.getYLabel()+"\n");
			count++;
		}
		writer1.close();
		writer2.close();
	}
	
	public void printPlotData(String simFile, String dissimFile) throws FileNotFoundException{
		ArrayList<_Doc> docs = m_corpus.getCollection();
		for(int i = 0; i < docs.size(); i++){
			for(int j = i+1; j < docs.size(); j++){
				_Doc d1 = docs.get(i);
				_Doc d2 = docs.get(j);
				if(d1.getYLabel() == d2.getYLabel())
					m_similar.add(Math.exp(-Utils.calculateCosineSimilarity(d1, d2)));
				else
					m_dissimilar.add(Math.exp(-Utils.calculateCosineSimilarity(d1, d2)));
			}
		}
//		//Define a descending comparator.
//		Comparator<Double> descending = new Comparator<Double>(){
//			public int compare(Double a, Double b){
//				if (a > b) return -1;
//				else if (a == b) return 0;
//				else return 1;
//			}
//		};
		Collections.sort(m_similar);
		Collections.sort(m_dissimilar);
//		//reverse the elements in m_similar.
//		for(int i = 0; i < m_similar.size() / 2; i++){
//			double tmp = m_similar.get(i);
//			m_similar.set(i, m_similar.get(m_similar.size() - 1 - i));
//			m_similar.set(m_similar.size() - 1 - i, tmp);
//		}
//		//reverse the elements in m_dissimilar.
//		for(int i = 0; i < m_dissimilar.size() / 2; i++){
//			double tmp = m_dissimilar.get(i);
//			m_dissimilar.set(i, m_dissimilar.get(m_dissimilar.size() - 1 - i));
//			m_dissimilar.set(m_dissimilar.size() - 1 - i, tmp);
//		}
		PrintWriter writer1 = new PrintWriter(new File(simFile));
		for(int i = 0; i < m_similar.size(); i=i+20){//take one sample every 20 points 
			double percentage = (double)(i+1) / m_similar.size();
			writer1.write(percentage + "," + m_similar.get(i) + "\n");
		}
		writer1.close();
		PrintWriter writer2 = new PrintWriter(new File(dissimFile));
		for(int i = 0; i < m_dissimilar.size(); i=i+20){//take one sample every 20 points 
			double percentage = (double)(i+1) / m_dissimilar.size();
			writer2.write(percentage + "," + m_dissimilar.get(i) + "\n");
		}
		writer2.close();
	}
	
	public void printPlotDataDiffClasses(String path) throws FileNotFoundException{
		ArrayList<ArrayList<_Doc>> groupedDocs = new ArrayList<ArrayList<_Doc>>();
		//ArrayList<ArrayList<Double>> groupedSimilarity = new ArrayList<ArrayList<Double>>();
		
		for(int i = 0; i < m_classNo; i++)
			groupedDocs.add(new ArrayList<_Doc>());
		
//		for(int i = 0; i < classNo * classNo; i++)
//			groupedSimilarity.add(i, new ArrayList<Double>());
		
		//Group all the documents into different classes.
		for(_Doc d: m_corpus.getCollection()){ 
			if(d.getYLabel() == 0)	groupedDocs.get(0).add(d);
			else if (d.getYLabel() == 1)	groupedDocs.get(1).add(d);
			else if (d.getYLabel() == 2)	groupedDocs.get(2).add(d);
			else if (d.getYLabel() == 3) 	groupedDocs.get(3).add(d);
			else if (d.getYLabel() == 4) 	groupedDocs.get(4).add(d);
			else 	System.err.println("Beyond the class number!!");
		}
		//Enumerate all similarities among all pairs.
		for(int i = 0; i < m_classNo; i++){//i is index.
			ArrayList<_Doc> tmp1 = groupedDocs.get(i);//Get the first set of docs.
			for(int j = 0; j < m_classNo; j++){
				if(i == j){
					//groupedSimilarity.add(i * classNo + j, calculateSimiPairs(tmp1));
					printSimilarity(path, i, j, calculateSimiPairs(tmp1));
				} else{
					ArrayList<_Doc> tmp2 = groupedDocs.get(j);//Get the second set of docs.
					//groupedSimilarity.add(i * classNo + j, calculateDissimiPairs(tmp1, tmp2));
					printSimilarity(path, i, j, calculateDissimiPairs(tmp1, tmp2));
				}
			}
		}
	}
	public void printSimilarity(String path, int m, int n, ArrayList<Double> similarities) throws FileNotFoundException{
		String fileName = path + m + "_" + n + "_small.csv";
		PrintWriter writer1 = new PrintWriter(new File(fileName));
		for(int i = 0; i < similarities.size(); i++){//take one sample every 20 points 
			double percentage = (double)(i+1) / similarities.size();
			writer1.write(percentage + "," + similarities.get(i) + "\n");
		}
		writer1.close();
	}
	//Calculate the similarities between similar pairs and return sorted similarities.
	public ArrayList<Double> calculateSimiPairs(ArrayList<_Doc> docs){
		ArrayList<Double> similarities = new ArrayList<Double>();
		for(int i = 0; i < docs.size(); i++){
			for(int j = i + 1; j < docs.size(); j++ ){
				similarities.add(Math.exp(-Utils.calculateCosineSimilarity(docs.get(i), docs.get(j))));
			}
		}
		Collections.sort(similarities);
		return similarities;
	}
	//Calculate the similarities between dissimilar pairs and return sorted similarities.
	public ArrayList<Double> calculateDissimiPairs(ArrayList<_Doc> tmp1, ArrayList<_Doc> tmp2){
		ArrayList<Double> similarities = new ArrayList<Double>();
		for(int i = 0; i < tmp1.size(); i++){
			for(int j = 0; j < tmp2.size(); j++ ){
				similarities.add(Math.exp(-Utils.calculateCosineSimilarity(tmp1.get(i), tmp2.get(j))));
			}
		}
		Collections.sort(similarities);
		return similarities;
	}
}
