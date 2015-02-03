package Classifier;

import influence.PageRank;

import java.io.IOException;
import java.text.ParseException;

import structures._Corpus;
import Analyzer.jsonAnalyzer;

public class AmazonReviewMain {

	public static void main(String[] args) throws IOException, ParseException{
		/*****Set these parameters before run the classifiers.*****/
		int featureSize = 0; //Initialize the fetureSize to be zero at first.
		int classNumber = 5; //Define the number of classes in this Naive Bayes.
		int Ngram = 2; //The default value is unigram. 
		int lengthThreshold = 5; //Document length threshold
		
		//"TF", "TFIDF", "BM25", "PLN"
		String featureValue = "BM25"; //The way of calculating the feature value, which can also be "TFIDF", "BM25"
		int norm = 2;//The way of normalization.(only 1 and 2)
		int CVFold = 10; //k fold-cross validation
		
		//"NB", "LR", "SVM", "PR"
		String classifier = "LR"; //Which classifier to use.
		
		//"SUP", "TRANS"
		String style = "TRANS";
		
		System.out.println("------------------------------------------------------------------------------------------------");
		System.out.println("Parameters of this run:" + "\nClassNumber: " + classNumber + "\tNgram: " + Ngram + "\tFeatureValue: " + featureValue + "\tLearning Method: " + style + "\tClassifier: " + classifier + "\nCross validation: " + CVFold);

		/*****The parameters used in loading files.*****/
		String folder = "./data/amazon/test01";
		String suffix = ".json";
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
		String finalLocation = "./data/Features/selected_fv_stat.txt";
		String featureLocation = "./data/Features/selected_fv.txt";
		String modelPath = "./data/Model/";

		/*****Parameters in feature selection.*****/
		String stopwords = "./data/Model/stopwords.dat";
		String featureSelection = "CHI"; //Feature selection method.
		double startProb = 0.3; // Used in feature selection, the starting point of the features.
		double endProb = 0.999; // Used in feature selection, the ending point of the features.
		int DFthreshold = 20; // Filter the features with DFs smaller than this threshold.
		System.out.println("Feature Seleciton: " + featureSelection + "\tStarting probability: " + startProb + "\tEnding probability:" + endProb);
		
		/*****Parameters in time series analysis.*****/
		int window = 3;
		System.out.println("Window length: " + window);
		System.out.println("------------------------------------------------------------------------------------------------");
		
		/*****Parameters in time series analysis.*****/
		//String debugOutput = "./data/debug/LR.output";
		
		/*****The staring point and ending point of cross validation.*****/
		//If (0-0.5), then we take the head part; if (0.5-1), we take the tail part.
//		double spTrain = 0.2; //The start percentage of the train set. 
//		double epTrain = 1.0; //The end percentage of the train set.
		
		//This part is used to verify the relationship between the number of reviews and ratings.		
//		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber, featureLocation, Ngram, lengthThreshold);
//		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
//		String excelPath = "./data/ReviewRating.xls";
//		analyzer.saveReviewRating(excelPath);
	
		
		/****Pre-process the data.*****/
		//Feture selection.
		System.out.println("Performing feature selection, wait...");
		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber, "", Ngram, lengthThreshold);
		analyzer.LoadStopwords(stopwords);
		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
		analyzer.featureSelection(featureLocation, featureSelection, startProb, endProb, DFthreshold); //Select the features.
		
		//Collect vectors for documents.
		System.out.println("Creating feature vectors, wait...");
//		jsonAnalyzer 
		analyzer = new jsonAnalyzer(tokenModel, classNumber, featureLocation, Ngram, lengthThreshold);
		//analyzer.setReleaseContent( !(classifier.equals("PR") || debugOutput!=null) );//Just for debugging purpose: all the other classifiers do not need content
		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
		analyzer.setFeatureValues(featureValue, norm);
		analyzer.setTimeFeatures(window);
		featureSize = analyzer.getFeatureSize();
		
		_Corpus corpus = analyzer.returnCorpus(finalLocation);
		double C = 0.1;
		
		//temporal code to add pagerank weights
//		PageRank tmpPR = new PageRank(corpus, classNumber, featureSize + window, C, 100, 50, 1e-6);
//		tmpPR.train(corpus.getCollection());
		
		/********Choose different classification methods.*********/
		//Execute different classifiers.
		if (style.equals("SUP")) {
			if(classifier.equals("NB")){
				//Define a new naive bayes with the parameters.
				System.out.println("Start naive bayes, wait...");
				NaiveBayes myNB = new NaiveBayes(corpus, classNumber, featureSize + window + 1);
				myNB.crossValidation(CVFold, corpus);//Use the movie reviews for testing the codes.
				
			} else if(classifier.equals("LR")){
				//Define a new logistics regression with the parameters.
				System.out.println("Start logistic regression, wait...");
				LogisticRegression myLR = new LogisticRegression(corpus, classNumber, featureSize + window + 1, C);
				//myLR.setDebugOutput(debugOutput);
				System.out.format("---------------------------------------------------------------------\n");
				myLR.crossValidation(CVFold, corpus);
				//myLR.saveModel(modelPath + "LR.model");
			} else if(classifier.equals("SVM")){
				System.out.println("Start SVM, wait...");
				SVM mySVM = new SVM(corpus, classNumber, featureSize + window + 1, C);
				mySVM.crossValidation(CVFold, corpus);
				
			} else if (classifier.equals("PR")){
				System.out.println("Start PageRank, wait...");
				PageRank myPR = new PageRank(corpus, classNumber, featureSize + window + 1, C, 100, 50, 1e-6);
				myPR.train(corpus.getCollection());
				
			} else System.out.println("Classifier has not developed yet!");
		} else if (style.equals("TRANS")) {
			SemiSupervised mySemi = new SemiSupervised(corpus, classNumber, featureSize + window + 1, classifier);
			mySemi.crossValidation(CVFold, corpus);
			
		} else System.out.println("Learning paradigm has not developed yet!");
	}
}
