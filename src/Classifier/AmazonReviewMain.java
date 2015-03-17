package Classifier;

import influence.PageRank;

import java.io.IOException;
import java.text.ParseException;

import structures._Corpus;
import Analyzer.jsonAnalyzer;
import Classifier.semisupervised.GaussianFields;
import Classifier.supervised.LogisticRegression;
import Classifier.supervised.NaiveBayes;
import Classifier.supervised.SVM;

public class AmazonReviewMain {

	public static void main(String[] args) throws IOException, ParseException{
		/*****Set these parameters before run the classifiers.*****/
		int featureSize = 0; //Initialize the fetureSize to be zero at first.
		int classNumber = 5; //Define the number of classes in this Naive Bayes.
		int Ngram = 1; //The default value is bigram. 
		int lengthThreshold = 0; //Document length threshold
		
		//"TF", "TFIDF", "BM25", "PLN"
		String featureValue = "BM25"; //The way of calculating the feature value, which can also be "TFIDF", "BM25"
		int norm = 2;//The way of normalization.(only 1 and 2)
		int CVFold = 10; //k fold-cross validation
	
		//"SUP", "SEMI", "FV"
		String style = "FV";
		
		//"NB", "LR", "SVM", "PR"
		String classifier = "LR"; //Which classifier to use.
		double C = 0.1;
//		String modelPath = "./data/Model/";
		String debugOutput = "data/debug/LR.output";
		
		System.out.println("--------------------------------------------------------------------------------------");
		System.out.println("Parameters of this run:" + "\nClassNumber: " + classNumber + "\tNgram: " + Ngram + "\tFeatureValue: " + featureValue + "\tLearning Method: " + style + "\tClassifier: " + classifier + "\nCross validation: " + CVFold);

//		/*****Parameters in feature selection.*****/
		String featureSelection = "CHI"; //Feature selection method.
		String stopwords = "./data/Model/stopwords.dat";
		double startProb = 0.3; // Used in feature selection, the starting point of the features.
		double endProb = 0.999; // Used in feature selection, the ending point of the features.
		int DFthreshold = 10; // Filter the features with DFs smaller than this threshold.
		System.out.println("Feature Seleciton: " + featureSelection + "\tStarting probability: " + startProb + "\tEnding probability:" + endProb);
		
		/*****The parameters used in loading files.*****/
		String diffFolder = "20json";
		String path = "data/" + diffFolder + "/";
		String folder = path + "RawData";
		String suffix = ".json";
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
		String tagModel = "./data/Model/en-pos-maxent.bin";
		String pattern = String.format("%dgram_%s_%s_%s", Ngram, featureValue, featureSelection, diffFolder);
		String featureLocation = String.format(path + "fv_%s.txt", pattern);//feature location
		//String featureStatLocation = String.format("data/fv_stat_%s.txt", pattern);//stat location
		String vctFile = String.format(path + "vct_%s.dat", pattern);		
		
		/*****Parameters in time series analysis.*****/
		int window = 0;
		System.out.println("Window length: " + window);
		System.out.println("--------------------------------------------------------------------------------------");
		
		/****Feture selection*****/
		System.out.println("Performing feature selection, wait...");
		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, tagModel, classNumber, "", Ngram, lengthThreshold);
		analyzer.LoadStopwords(stopwords);
		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
		analyzer.featureSelection(featureLocation, featureSelection, startProb, endProb, DFthreshold); //Select the features.
		
		//Collect vectors for documents.
		System.out.println("Creating feature vectors, wait...");
		//jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber, finalLocation, Ngram, lengthThreshold);
		analyzer.setReleaseContent( !(classifier.equals("PR") || debugOutput!=null) );//Just for debugging purpose: all the other classifiers do not need content
		analyzer.disablePosTagger();//set the m_tagger = null so that no postagging is performed when load documents again.
		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
		analyzer.setFeatureValues(featureValue, norm);
		analyzer.setTimeFeatures(window);
		
		String xFile = path + diffFolder + "X.csv";
		String yFile = path + diffFolder + "Y.csv";
		featureSize = analyzer.getFeatureSize();
		analyzer.printXY(xFile, yFile);
		_Corpus corpus = analyzer.getCorpus();
		
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
				myLR.setDebugOutput(debugOutput);
				
				myLR.crossValidation(CVFold, corpus);//Use the movie reviews for testing the codes.
				//myLR.saveModel(modelPath + "LR.model");
			} else if(classifier.equals("SVM")){
				System.out.println("Start SVM, wait...");
				SVM mySVM = new SVM(corpus, classNumber, featureSize + window + 1, C, 0.01);//default eps value from Lin's implementation
				mySVM.crossValidation(CVFold, corpus);
				
			} else if (classifier.equals("PR")){
				System.out.println("Start PageRank, wait...");
				PageRank myPR = new PageRank(corpus, classNumber, featureSize + window + 1, C, 100, 50, 1e-6);
				myPR.train(corpus.getCollection());
				
			} else System.out.println("Classifier has not developed yet!");
		} else if (style.equals("SEMI")) {
			GaussianFields mySemi = new GaussianFields(corpus, classNumber, featureSize + window + 1, classifier);
			mySemi.crossValidation(CVFold, corpus);
		} else if (style.equals("FV")) {
			corpus.save2File(vctFile);
			System.out.format("Vectors saved to %s...\n", vctFile);
		} else System.out.println("Learning paradigm has not developed yet!");
	}
}
