package Classifier;

import influence.PageRank;

import java.io.IOException;
import java.text.ParseException;

import structures._Corpus;
import Analyzer.jsonAnalyzer;
import Classifier.metricLearning.LinearSVMMetricLearning;
import Classifier.semisupervised.GaussianFields;
import Classifier.semisupervised.GaussianFieldsByRandomWalk;
import Classifier.supervised.KNN;
import Classifier.supervised.LogisticRegression;
import Classifier.supervised.NaiveBayes;
import Classifier.supervised.SVM;

public class AmazonReviewMain {

	public static void main(String[] args) throws IOException, ParseException{
		/*****Set these parameters before running the classifiers.*****/
		int featureSize = 0; //Initialize the fetureSize to be zero at first.
		int Ngram = 1; //The default value is bigram. 
		
		//"TF", "TFIDF", "BM25", "PLN"
		String featureValue = "BM25"; //The way of calculating the feature value, which can also be "TFIDF", "BM25"
		int norm = 2;//The way of normalization.(only 1 and 2)
		
		int classNumber = 5; //Define the number of classes in this Naive Bayes.
		int lengthThreshold = 10; //Document length threshold
		int CVFold = 10; //k fold-cross validation

		//"SUP", "SEMI", "FV: save features and vectors to files"
		String style = "FV";//"SUP", "SEMI"
		//Supervised: "NB", "LR", "PR-LR", "SVM"; Semi-supervised: "GF", "GF-RW", "GF-RW-ML"**/
		String classifier = "KNN"; //Which classifier to use.
		String multipleLearner = "SVM";
		double C = 1.0;
		
//		String modelPath = "./data/Model/";
		System.out.println("--------------------------------------------------------------------------------------");
		System.out.println("Parameters of this run:" + "\nClassNumber: " + classNumber + "\tNgram: " + Ngram + "\tFeatureValue: " + featureValue + "\tLearning Method: " + style + "\tClassifier: " + classifier + "\nCross validation: " + CVFold);

		/*****Parameters in feature selection.*****/
		String featureSelection = "CHI"; //Feature selection method.
		String stopwords = "./data/Model/stopwords.dat";
		double startProb = 0.0; // Used in feature selection, the starting point of the features.
		double endProb = 0.999; // Used in feature selection, the ending point of the features.
		int DFthreshold = 5; // Filter the features with DFs smaller than this threshold.
		System.out.println("Feature Seleciton: " + featureSelection + "\tStarting probability: " + startProb + "\tEnding probability:" + endProb);
		
		/*****The parameters used in loading files.*****/
		String diffFolder = "small";
		String path = "data/" + diffFolder + "/";
		String folder = path + "RawData";
		String suffix = ".json";
		
		String pattern = String.format("%dgram_%s_%s_%s", Ngram, featureValue, featureSelection, diffFolder);
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
		String featureLocation = String.format(path + "fv_%s.txt", pattern);//feature location
		String vctFile = String.format(path + "vct_%s.dat", pattern);
		String debugOutput = path + classifier + "_noPOS.csv";
		
		/*****Parameters in time series analysis.*****/
		int window = 0;
		System.out.println("Window length: " + window);
		System.out.println("--------------------------------------------------------------------------------------");
		
		/****Feature selection*****/
//		System.out.println("Performing feature selection, wait...");
//		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber, "", Ngram, lengthThreshold);
//		analyzer.LoadStopwords(stopwords);
//		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
//		analyzer.featureSelection(featureLocation, featureSelection, startProb, endProb, DFthreshold); //Select the features.
//		analyzer.resetStopwords();
		
		/****Create feature vectors*****/
		System.out.println("Creating feature vectors, wait...");
		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber,featureLocation, Ngram, lengthThreshold);
		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
		analyzer.setFeatureValues(featureValue, norm);
		analyzer.setTimeFeatures(window);
		
		_Corpus corpus = analyzer.getCorpus();
		featureSize = analyzer.getFeatureSize();
		
		/**Paramters in KNN.**/
//		int[] kArray = {2, 3, 4, 5, 6};
//		int[] lArray = {1, 2, 3, 4, 5, 6, 7, 8 ,9, 10};
//		corpus.save2File(vctFile);
//		String plotFile = path + "pairData_" + lengthThreshold + ".dat";
//		analyzer.printPlotData2OneFile(plotFile);
		
		//String matrixFile = path + "matrixA0321.dat";
//		/***Print the matrix of X and Y for metric learning.***/
//		String xFile = path + diffFolder + "X.csv";
//		String yFile = path + diffFolder + "Y.csv";
//		analyzer.printXY(xFile, yFile);
		
		//temporal code to add pagerank weights
//		PageRank tmpPR = new PageRank(corpus, classNumber, featureSize + window, C, 100, 50, 1e-6);
//		tmpPR.train(corpus.getCollection());
		
		/********Choose different classification methods.*********/
		if (style.equals("SUP")) {
			if(classifier.equals("NB")){
				//Define a new naive bayes with the parameters.
				System.out.println("Start naive bayes, wait...");
				NaiveBayes myNB = new NaiveBayes(corpus, classNumber, featureSize);
				myNB.crossValidation(CVFold, corpus);//Use the movie reviews for testing the codes.
				
			} else if(classifier.equals("LR")){
				//Define a new logistics regression with the parameters.
				System.out.println("Start logistic regression, wait...");
				LogisticRegression myLR = new LogisticRegression(corpus, classNumber, featureSize, C);
				myLR.setDebugOutput(debugOutput);//Save debug information into file.
				myLR.crossValidation(CVFold, corpus);//Use the movie reviews for testing the codes.
				//myLR.saveModel(modelPath + "LR.model");
			} else if(classifier.equals("SVM")){
				//Define a new SVM with the parameters.
				System.out.println("Start SVM, wait...");
				SVM mySVM = new SVM(corpus, classNumber, featureSize, C, 0.01);//default eps value from Lin's implementation
				mySVM.crossValidation(CVFold, corpus);
				
			} else if (classifier.equals("PR")){
				//Define a new Pagerank with parameters.
				System.out.println("Start PageRank, wait...");
				PageRank myPR = new PageRank(corpus, classNumber, featureSize, C, 100, 50, 1e-6);
				myPR.train(corpus.getCollection());
				
			} else if(classifier.equals("KNN")){
				System.out.println("Start KNN, wait...");
//				for(int k: kArray){
//					for(int l: lArray){
						int k=10, l=1;
						System.out.print(String.format("k=%d, l=%d\n", k, l));
						KNN myKNN = new KNN(corpus, classNumber, featureSize, k, l);
						myKNN.crossValidation(CVFold, corpus);
//					}
//				}
			} else 
				System.out.println("Classifier has not developed yet!");
		}
		else if (style.equals("SEMI")) {
			if (classifier.equals("GF")) {
				GaussianFields mySemi = new GaussianFields(corpus, classNumber, featureSize, multipleLearner);
				mySemi.crossValidation(CVFold, corpus);
			} else if (classifier.equals("GF-RW")) {
				GaussianFields mySemi = new GaussianFieldsByRandomWalk(corpus, classNumber, featureSize, multipleLearner, 1, 1, 5, 1, 0.1, 1e-4, 1, false);
				mySemi.setFeaturesLookup(analyzer.getFeaturesLookup()); //give the look up to the classifer for debugging purpose.
//				mySemi.setDebugOutput(debugOutput);
				//mySemi.setMatrixA(analyzer.loadMatrixA(matrixFile));
				mySemi.crossValidation(CVFold, corpus);
			} else if (classifier.equals("GF-RW-ML")) {
				LinearSVMMetricLearning lMetricLearner = new LinearSVMMetricLearning(corpus, classNumber, featureSize, multipleLearner, 0.1, 100, 50, 1.0, 0.1, 1e-4, 0.1, false, 3, 0.01);
				lMetricLearner.setDebugOutput(debugOutput);
				lMetricLearner.crossValidation(CVFold, corpus);
			} else System.out.println("Classifier has not been developed yet!");
		} else if (style.equals("FV")) {
			corpus.save2File(vctFile);
			System.out.format("Vectors saved to %s...\n", vctFile);
		} else 
			System.out.println("Learning paradigm has not developed yet!");
	}
}
