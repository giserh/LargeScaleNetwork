package Classifier;

import influence.PageRank;

import java.io.IOException;
import java.text.ParseException;

import structures._Corpus;
import Analyzer.VctAnalyzer;
import Classifier.metricLearning.LinearSVMMetricLearning;
import Classifier.semisupervised.GaussianFields;
import Classifier.semisupervised.GaussianFieldsByRandomWalk;
import Classifier.supervised.KNN;
import Classifier.supervised.LogisticRegression;
import Classifier.supervised.NaiveBayes;
import Classifier.supervised.PRLogisticRegression;
import Classifier.supervised.SVM;

public class VectorReviewMain {

	public static void main(String[] args) throws IOException, ParseException{
		/*****Set these parameters before run the classifiers.*****/
		int classNumber = 5; //Define the number of classes in this Naive Bayes.
		int lengthThreshold = 0; //Document length threshold
		int CVFold = 10; //k fold-cross validation

		//"SUP", "SEMI", "FV: save features and vectors to files"
		String style = "SEMI";//"SUP", "SEMI"
		//Supervised: "NB", "LR", "PR-LR", "SVM"; Semi-supervised: "GF", "GF-RW", "GF-RW-ML"**/
		String classifier = "GF-RW"; //Which classifier to use.
		String multipleLearner = "SVM";
		double C = 1.0;		

		/*****The parameters used in loading files.*****/
		String diffFolder = "small";
		String path = "data/" + diffFolder + "/";
		String featureLocation = String.format("%sfv_1gram_BM25_CHI_%s.txt", path, diffFolder);
		String vctfile = String.format("%svct_1gram_BM25_CHI_%s.dat", path, diffFolder);
		
		String matrixFile = path + "matrixA.dat";
		
		/****Pre-process the data.*****/
		System.out.println("Loading vectors from file, wait...");
		VctAnalyzer analyzer = new VctAnalyzer(classNumber, lengthThreshold, featureLocation);
		analyzer.LoadDoc(vctfile); //Load all the documents as the data set.
		
		/***The parameters used in GF-RW and debugging.****/
		double eta = 0.1, sr = 1;
		String debugOutput = path + classifier + eta + "_noPOS.txt";
		String WrongRWfile= path + classifier + eta + "_WrongRW.txt";
		String WrongSVMfile= path + classifier + eta + "_WrongSVM.txt";
		String FuSVM = path + classifier + eta + "_FuSVMResults.txt";
		
//		//We can also print the matrix of X and Y with vectors.
//		String xFile = path + diffFolder + "X.csv";
//		String yFile = path + diffFolder + "Y.csv";
//		analyzer.printXY(xFile, yFile);
		
		_Corpus corpus = analyzer.getCorpus();
		int featureSize = corpus.getFeatureSize();
		
		/**Paramters in KNN.**/
		int k = 1, l = 0;//l > 0, random projection; else brute force.
		
//		//Print the distance vs similar(dissimilar pairs)
//		String simFile = path + "similarPlot.csv";
//		String dissimFile = path + "dissimilarPlot.csv";
//		analyzer.printPlotData(simFile, dissimFile);
		
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
				myLR.crossValidation(CVFold, corpus);//Use the movie reviews for testing the codes.
				//myLR.saveModel(modelPath + "LR.model");
				
			} else if(classifier.equals("PRLR")){
				//Define a new logistics regression with the parameters.
				System.out.println("Start posterior regularized logistic regression, wait...");
				PRLogisticRegression myLR = new PRLogisticRegression(corpus, classNumber, featureSize, C);
				myLR.setDebugOutput(debugOutput);
				myLR.crossValidation(CVFold, corpus);//Use the movie reviews for testing the codes.
				//myLR.saveModel(modelPath + "LR.model");
				
			} else if(classifier.equals("SVM")){
				System.out.println("Start SVM, wait...");
				SVM mySVM = new SVM(corpus, classNumber, featureSize, C, 0.001);//default value of eps from Lin's implementation
				mySVM.crossValidation(CVFold, corpus);
				
			} else if (classifier.equals("PR")){
				System.out.println("Start PageRank, wait...");
				PageRank myPR = new PageRank(corpus, classNumber, featureSize, C, 100, 50, 1e-6);
				myPR.train(corpus.getCollection());
			
			} else if(classifier.equals("KNN")){
				System.out.println(String.format("Start KNN, k=%d, l=%d, wait...", k, l));
				KNN myKNN = new KNN(corpus, classNumber, featureSize, k, l);
				myKNN.crossValidation(CVFold, corpus);
			}
			else System.out.println("Classifier has not been developed yet!");
		} else if (style.equals("SEMI")) {
			if (classifier.equals("GF")) {
				GaussianFields mySemi = new GaussianFields(corpus, classNumber, featureSize, multipleLearner);
				mySemi.crossValidation(CVFold, corpus);
			} else if (classifier.equals("GF-RW")) {
//				GaussianFields mySemi = new GaussianFieldsByRandomWalk(corpus, classNumber, featureSize, multipleLearner, 1, 1, 5, 1, 0, 1e-4, 1, false);
				GaussianFields mySemi = new GaussianFieldsByRandomWalk(corpus, classNumber, featureSize, multipleLearner, sr, 100, 50, 1, 0.1, 1e-4, eta, false);
				mySemi.setFeaturesLookup(analyzer.getFeaturesLookup()); //give the look up to the classifier for debugging purpose.
				mySemi.setDebugOutput(debugOutput);
				mySemi.setDebugPrinters(WrongRWfile, WrongSVMfile, FuSVM);
//				mySemi.setMatrixA(analyzer.loadMatrixA(matrixFile));
				mySemi.crossValidation(CVFold, corpus);
			} else if (classifier.equals("GF-RW-ML")) {
				LinearSVMMetricLearning lMetricLearner = new LinearSVMMetricLearning(corpus, classNumber, featureSize, multipleLearner, 0.1, 100, 50, 1.0, 0.1, 1e-4, 0.1, false, 3, 0.01);
				lMetricLearner.setDebugOutput(debugOutput);
				lMetricLearner.crossValidation(CVFold, corpus);
			} else System.out.println("Classifier has not been developed yet!");
			
		} else System.out.println("Learning paradigm has not been developed yet!");
	} 
}
