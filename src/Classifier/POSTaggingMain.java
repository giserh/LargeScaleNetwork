package Classifier;

import influence.PageRank;
import java.io.IOException;
import java.text.ParseException;
import structures._Corpus;
import Analyzer.jsonAnalyzer;
import Classifier.metricLearning.LinearSVMMetricLearning;
import Classifier.semisupervised.GaussianFields;
import Classifier.semisupervised.GaussianFieldsByRandomWalk;
import Classifier.supervised.LogisticRegression;
import Classifier.supervised.NaiveBayes;
import Classifier.supervised.SVM;

public class POSTaggingMain {

	public static void main(String[] args) throws IOException, ParseException{
		/*****Set these parameters before run the classifiers.*****/
		int featureSize = 0; //Initialize the fetureSize to be zero at first.
		int classNumber = 5; //Define the number of classes in this Naive Bayes.
		int Ngram = 1; //The default value is unigram for pos tagging. 
		int lengthThreshold = 10; //Document length threshold
		
		//"TF", "TFIDF", "BM25", "PLN"
		String featureValue = "BM25"; //The way of calculating the feature value, which can also be "TFIDF", "BM25"
		int norm = 2;//The way of normalization.(only 1 and 2)
		int CVFold = 10; //k fold-cross validation
	
		//"SUP", "SEMI", "FV: save features and vectors to files"
		String style = "SEMI";//"SUP", "SEMI"
		//Supervised: "NB", "LR", "PR-LR", "SVM"; Semi-supervised: "GF", "GF-RW", "GF-RW-ML"**/
		String classifier = "GF-RW"; //Which classifier to use.
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
		String featureLocation = String.format(path + "fv_%s.txt", pattern);//feature location

		String tokenModel = "./data/Model/en-token.bin"; //Token model.
		String stnModel = "./data/Model/en-sent.bin"; //Sentence model.
		String tagModel = "./data/Model/en-pos-maxent.bin";		
		
		/*****Parameters in time series analysis.*****/
		int window = 0;
		System.out.println("Window length: " + window);
		
		/****Parameters related with POS Tagging.***/
		int posTaggingMethod = 4; //Which way to use to build features with pos tagging.
		String SNWfile = "data/Model/SentiWordNet_3.0.0_20130122.txt";
		System.out.format("Postagging method: %d\n", posTaggingMethod);
		
		/***The parameters used in GF-RW and debugging.****/
		double eta = 0.1, sr = 1;
		String debugOutput = path + classifier + "_POS" + posTaggingMethod + ".txt";
		String WrongRWfile= path + classifier + eta + "_POS" + posTaggingMethod + "_WrongRW.txt";
		String WrongSVMfile= path + classifier + eta + "_POS" + posTaggingMethod + "_WrongSVM.txt";
		String FuSVM = path + classifier + eta + "_POS" + posTaggingMethod + "_FuSVMResults.txt";
		
		/***The parameter used in the sentiwordnet analysis.***/
		String scoreFile = path + classifier + "_POS" + posTaggingMethod + "_SentiWordNetScore.txt";
		String projFeatureFile = path + classifier + "_POS" + posTaggingMethod + "_OverlapFeatures.txt";
		
		//With the given CV, build the projected vectors for all documents.
		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, stnModel, tagModel, classNumber, featureLocation, Ngram, lengthThreshold, posTaggingMethod);
		if( posTaggingMethod == 3) // Load the SNW file first.
			analyzer.LoadSNW(SNWfile);
		
//		if( posTaggingMethod == 4) { // Load the SNW with scores.
//			int k = 10;
//			analyzer.LoadSNWWithScore(SNWfile);
//			analyzer.saveSentiWordNetFeatures(scoreFile);
//			analyzer.setFeatureDimension(k);
//		}
		
		if( posTaggingMethod == 4) { // Load the SNW with scores.
			int k = 10;
			analyzer.LoadSNWWithScore(SNWfile);
			analyzer.LoadProjFeaturesWithScores(projFeatureFile);
//			analyzer.saveSentiWordNetFeatures(scoreFile);
			analyzer.setFeatureDimension(k);
			analyzer.AssignFeatureIndexes();
		}

		analyzer.LoadDirectory(folder, suffix); //Load all the documents to build the sparse vectors and projected vectors.
		analyzer.setFeatureValues(featureValue, norm);
		analyzer.setTimeFeatures(window);
		
//		if(posTaggingMethod == 4 )
//			analyzer.saveProjFeaturesScores(projFeatureFile);
		
		featureSize = analyzer.getFeatureSize();
		_Corpus corpus = analyzer.getCorpus();
		String vctFile = String.format(path + "vct_%s.dat", pattern);
		String projectedVctFile = String.format(path + "vct_projected_%s.dat", pattern);

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
				
			} else System.out.println("Classifier has not developed yet!");
		}
		else if (style.equals("SEMI")) {
			if (classifier.equals("GF")) {
				GaussianFields mySemi = new GaussianFields(corpus, classNumber, featureSize, multipleLearner);
				mySemi.crossValidation(CVFold, corpus);
			} else if (classifier.equals("GF-RW")) {
				GaussianFields mySemi = new GaussianFieldsByRandomWalk(corpus, classNumber, featureSize, multipleLearner, sr, 100, 50, 1.0, 0.1, 1e-4, eta, false);
				//With pos tagging, we need the look-up table for projected features.
				mySemi.setFeaturesLookup(analyzer.getProjFeaturesLookup());
				mySemi.setPOSTagging(posTaggingMethod);
				mySemi.setDebugOutput(debugOutput);
				mySemi.setDebugPrinters(WrongRWfile, WrongSVMfile, FuSVM);
				//mySemi.setMatrixA(analyzer.loadMatrixA(matrixFile));
				mySemi.crossValidation(CVFold, corpus);
			} else if (classifier.equals("GF-RW-ML")) {
				LinearSVMMetricLearning lMetricLearner = new LinearSVMMetricLearning(corpus, classNumber, featureSize, multipleLearner, 0.1, 100, 50, 1.0, 0.1, 1e-4, 0.1, false, 3, 0.01);
				lMetricLearner.setDebugOutput(debugOutput);
				lMetricLearner.crossValidation(CVFold, corpus);
			} else System.out.println("Classifier has not been developed yet!");
		} else if (style.equals("FV")) {
			corpus.save2File(vctFile);
			corpus.save2FileProjectSpVct(projectedVctFile);
			System.out.format("Vectors saved to %s...\n", vctFile);
		} else 
			System.out.println("Learning paradigm has not developed yet!");
	}
}
