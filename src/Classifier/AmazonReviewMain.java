package Classifier;

import influence.PageRank;

import java.io.IOException;
import java.text.ParseException;

import structures._Corpus;
import Analyzer.jsonAnalyzer;

public class AmazonReviewMain {

	public static void main(String[] args) throws IOException, ParseException{
		/*****Set these parameters before run the classifiers.*****/
		int classNumber = 5; //Define the number of classes in this Naive Bayes.
		int Ngram = 2; //The default value is bigram. 
		int lengthThreshold = 5; //Document length threshold
		
		//"TF", "TFIDF", "BM25", "PLN"
		String featureValue = "TF"; //The way of calculating the feature value, which can also be "TFIDF", "BM25"
		int norm = 2;//The way of normalization.(only 1 and 2)
		int CVFold = 10; //k fold-cross validation
		
		//"NB", "LR", "SVM", "PR"
		String supModel = "LR"; //Which classifier to use.
		
		//"SUP", "TRANS", "TM"
		String style = "TRANS";
		//"SM", "RW", "SG"
		String transModel = "RW";
		//?????
		String topicModel = "";
		//sampleRate, kUL, kUU, TLalpha, TLbeta are defined in SemiSupervised.java
		System.out.println("------------------------------------------------------------------------------------------------");
		System.out.println("Parameters of this run:" + "\nClassNumber: " + classNumber + "\tNgram: " + Ngram + "\tFeatureValue: " + featureValue + "\tLearning Method: " + style + "\tClassifier: " + supModel + "\nCross validation: " + CVFold);

		/*****The parameters used in loading files.*****/
		String folder = "./data/amazon/test01";
		String suffix = ".json";
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
		String stnModel = "./data/Model/en-sent.bin"; //Sentence model.
		String stopwords = "./data/Model/stopwords.dat";
		String featureFile = null;//list of controlled vocabulary
		String featureStat= null;//detailed statistics of the selected features
		//String modelPath = "./data/Model/";

		/*****Parameters in feature selection.*****/
		String featureSelection = "CHI"; //Feature selection method.
		double startProb = 0.4; // Used in feature selection, the starting point of the features.
		double endProb = 0.999; // Used in feature selection, the ending point of the features.
		int DFthreshold = 10; // Filter the features with DFs smaller than this threshold.
		System.out.println("Feature Seleciton: " + featureSelection + "\tStarting probability: " + startProb + "\tEnding probability:" + endProb);
		
		/*****Parameters in time series analysis.*****/
		int window = 0;
		
		System.out.println("Window length: " + window);
		System.out.println("------------------------------------------------------------------------------------------------");
		
		/*****Parameters in time series analysis.*****/
		String debugOutput = "./data/debug/LR.output";
		
		/*****Parameters specified for classifiers.*****/
		double C = 0.1; // trade-off parameter in LR and SVM
		
		/*****The staring point and ending point of cross validation.*****/
		//If (0-0.5), then we take the head part; if (0.5-1), we take the tail part.
//		double spTrain = 0.2; //The start percentage of the train set. 
//		double epTrain = 1.0; //The end percentage of the train set.
		
//		//This part is used to verify the relationship between the number of reviews and ratings.		
//		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber, featureLocation, Ngram, lengthThreshold);
//		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
//		String excelPath = "./data/ReviewRating.xls";
//		analyzer.saveReviewRating(excelPath);
		
		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber, "", Ngram, lengthThreshold);
		if(featureFile == null){
			/****Pre-process the data.*****/
			//Feture selection.
			System.out.println("Performing feature selection, wait...");
			featureFile = String.format("./data/Features/%s_fv.dat", featureSelection);
			featureStat = String.format("./data/Features/%s_fv_stat.dat", featureSelection);
			analyzer.LoadStopwords(stopwords);
			analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
			analyzer.featureSelection(featureFile, featureSelection, startProb, endProb, DFthreshold); //Select the features.
			//analyzer.reset();
		}
		
//		//Collect vectors for documents.
		System.out.println("Creating feature vectors, wait...");
		analyzer = new jsonAnalyzer(tokenModel, classNumber, featureFile, Ngram, lengthThreshold);
//		analyzer.setReleaseContent( !(model.equals("PR") || debugOutput!=null) );//Just for debugging purpose: all the other classifiers do not need content
		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
		analyzer.setFeatureValues(featureValue, norm);
		analyzer.setTimeFeatures(window);
		
		int featureSize = analyzer.getFeatureSize();
		_Corpus corpus = analyzer.returnCorpus(featureStat);
		
//		temporal code to add pagerank weights
//		PageRank tmpPR = new PageRank(corpus, classNumber, featureSize + window, C, 100, 50, 1e-6);
//		tmpPR.train(corpus.getCollection());
		
		/********Choose different classification methods.*********/
		//Execute different classifiers.
		if (style.equals("SUP")) {
			if(supModel.equals("NB")){
				//Define a new naive bayes with the parameters.
				System.out.println("Start naive bayes, wait...");
				NaiveBayes myNB = new NaiveBayes(corpus, classNumber, featureSize + window + 1);
				myNB.crossValidation(CVFold, corpus);//Use the movie reviews for testing the codes.
				
			} else if(supModel.equals("LR")){
				//Define a new logistics regression with the parameters.
				System.out.println("Start logistic regression, wait...");
				LogisticRegression myLR = new LogisticRegression(corpus, classNumber, featureSize + window + 1, C);
				//myLR.setDebugOutput(debugOutput);
				System.out.format("---------------------------------------------------------------------\n");
				myLR.crossValidation(CVFold, corpus);
				//myLR.saveModel(modelPath + "LR.model");
			} else if(supModel.equals("SVM")){
				System.out.println("Start SVM, wait...");
				SVM mySVM = new SVM(corpus, classNumber, featureSize + window + 1, C);
				mySVM.crossValidation(CVFold, corpus);
				
			} else if (supModel.equals("PR")){
				System.out.println("Start PageRank, wait...");
				PageRank myPR = new PageRank(corpus, classNumber, featureSize + window + 1, C, 100, 50, 1e-6);
				myPR.train(corpus.getCollection());
				
			} else System.out.println("This SUP classifier has not developed yet!");
		} else if (style.equals("TRANS")) {
			if(transModel.equals("SM")){
				System.out.println("Start SemiSupervised learning, wait...");
				SemiSupervised mySM = new SemiSupervised(corpus, classNumber, featureSize + window + 1, supModel);
				mySM.crossValidation(CVFold, corpus);
				
			} else if(transModel.equals("RW")){
				System.out.println("Start Semi Randow Walk, wait...");
				SemiRandomWalk myRW = new SemiRandomWalk(corpus, classNumber, featureSize + window + 1, supModel);
				myRW.crossValidation(CVFold, corpus);
				
			} else if(transModel.equals("SG")){
				System.out.println("Start Semi Gaussian learning, wait...");
				SemiGaussian mySG = new SemiGaussian(corpus, classNumber, featureSize + window + 1, supModel);
				mySG.crossValidation(CVFold, corpus);
				
			} else System.out.println("This TRANS classifier has not developed yet!!");
		} else System.out.println("Learning paradigm has not developed yet!");
	}
}
