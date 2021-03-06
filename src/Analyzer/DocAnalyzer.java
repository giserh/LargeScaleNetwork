package Analyzer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Set;
import java.util.TreeMap;
import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.InvalidFormatException;
import org.tartarus.snowball.SnowballStemmer;
import org.tartarus.snowball.ext.englishStemmer;
import structures._Doc;
import structures._RankItem;
import structures._SparseFeature;
import utils.Utils;

public class DocAnalyzer extends Analyzer {
	protected Tokenizer m_tokenizer;
	protected SnowballStemmer m_stemmer;
	protected SentenceDetectorME m_stnDetector;
	
	protected POSTaggerME m_tagger;
	protected Set<String> m_rawFilter; //Use this to store all the adj/advs for later filtering.
	protected HashMap<Integer, String> m_filter;//Use this to store all the indexes of adj/advs for use.
	Set<String> m_stopwords;
	protected boolean m_releaseContent;
	protected int m_posTaggingMethod;
	protected Set<String> m_dictionary;//The set for storing sentinet word.
    protected HashMap<String, Double> m_dictMap; //The map stores the strings and corrresponding scores.
    protected ArrayList<_RankItem> m_sentiwords;
    protected HashMap<String, Integer> m_sentiIndex;
    
	//Constructor with ngram and fValue.
	public DocAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold) throws InvalidFormatException, FileNotFoundException, IOException{
		super(classNo, threshold);
		m_tokenizer = new TokenizerME(new TokenizerModel(new FileInputStream(tokenModel)));
		m_stemmer = new englishStemmer();		
		m_stnDetector = null; // indicating we don't need sentence splitting
		
		m_Ngram = Ngram;
		m_isCVLoaded = LoadCV(providedCV);
		m_stopwords = new HashSet<String>();
		m_releaseContent = true;
	}
	
	//Constructor with ngram and fValue and sentence check.
	public DocAnalyzer(String tokenModel, String stnModel, int classNo, String providedCV, int Ngram, int threshold) throws InvalidFormatException, FileNotFoundException, IOException{
		super(classNo, threshold);
		m_tokenizer = new TokenizerME(new TokenizerModel(new FileInputStream(tokenModel)));
		m_stemmer = new englishStemmer();
		
		if (stnModel!=null)
			m_stnDetector = new SentenceDetectorME(new SentenceModel(new FileInputStream(stnModel)));
		else
			m_stnDetector = null;
		
		m_Ngram = Ngram;
		m_isCVLoaded = LoadCV(providedCV);
		m_stopwords = new HashSet<String>();
		m_releaseContent = true;		
	}
	
	//Constructor with ngram and fValue.
	public DocAnalyzer(String tokenModel, String stnModel, String tagModel, int classNo, String providedCV, int Ngram, int threshold, int posTaggingMethod) throws InvalidFormatException, FileNotFoundException, IOException {
		super(classNo, threshold);
		m_tokenizer = new TokenizerME(new TokenizerModel(new FileInputStream(tokenModel)));
		m_stemmer = new englishStemmer();		//We need to spilt sentences before we do pos tagging.
		if (stnModel!=null)
			m_stnDetector = new SentenceDetectorME(new SentenceModel(new FileInputStream(stnModel)));
		else
			m_stnDetector = null;
		
		m_tagger = new POSTaggerME(new POSModel(new FileInputStream(tagModel)));

		m_Ngram = Ngram;
		m_isCVLoaded = LoadCV(providedCV);
		m_stopwords = new HashSet<String>();
		m_rawFilter = new HashSet<String>(); //The adj/adv words used to construct the new features.
		m_filter = new HashMap<Integer, String>();
		m_releaseContent = true;
		m_posTaggingMethod = posTaggingMethod;
	}
	
	public void setReleaseContent(boolean release) {
		m_releaseContent = release;
	}
	
	public void LoadStopwords(String filename) {
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;

			while ((line = reader.readLine()) != null) {
				line = SnowballStemming(Normalize(line));//****
				if (!line.isEmpty())
					m_stopwords.add(line);
			}
			reader.close();
			System.out.format("Loading %d stopwords from %s", m_stopwords.size(), filename);
		} catch(IOException e){
			System.err.format("[Error]Failed to open file %s!!", filename);
		}
	}
	
	//Tokenizer.
	protected String[] Tokenizer(String source){
		String[] tokens = m_tokenizer.tokenize(source);
		return tokens;
	}
	
	//Normalize.
	protected String Normalize(String token){
		//token = Normalizer.normalize(token, Normalizer.Form.NFKC);
		token = token.replaceAll("\\W+", "");
		token = token.toLowerCase();
		
		if (Utils.isNumber(token))
			return "NUM";
		else
			return token;
	}
	
	//Snowball Stemmer.
	protected String SnowballStemming(String token){
		m_stemmer.setCurrent(token);
		if(m_stemmer.stem())
			return m_stemmer.getCurrent();
		else
			return token;
	}
	
//	public String PorterStemming(String token) {
////		porterStemmer stemmer = new porterStemmer();
//		m_stemmer.setCurrent(token);
//		if (m_stemmer.stem())
//			return m_stemmer.getCurrent();
//		else
//			return token;
//	}
	
	protected boolean isLegit(String token) {
		return !token.isEmpty() 
			&& !m_stopwords.contains(token)
			&& token.length()>1
			&& token.length()<20;
	}
	
	protected boolean isBoundary(String token) {
		return token.isEmpty();//is this a good checking condition?
	}
	
	//Given a long string, tokenize it, normalie it and stem it, return back the string array.
	protected String[] TokenizerNormalizeStemmer(String source){
		String[] tokens = Tokenizer(source); //Original tokens.
		//Normalize them and stem them.		
		for(int i = 0; i < tokens.length; i++)
			tokens[i] = SnowballStemming(Normalize(tokens[i]));
		
		LinkedList<String> Ngrams = new LinkedList<String>();
		int tokenLength = tokens.length, N = m_Ngram;		
		for(int i=0; i<tokenLength; i++) {
			String token = tokens[i];
			boolean legit = isLegit(token);
			if (legit)
				Ngrams.add(token);//unigram
			
			//N to 2 grams
			if (!isBoundary(token)) {
				for(int j=i-1; j>=Math.max(0, i-N+1); j--) {	
					if (isBoundary(tokens[j]))
						break;//touch the boundary
					
					token = tokens[j] + "-" + token;
					legit |= isLegit(tokens[j]);
					if (legit)//at least one of them is legitimate
						Ngrams.add(token);
				}
			}
		}
		
		return Ngrams.toArray(new String[Ngrams.size()]);
	}

	//Load a movie review document and analyze it.
	//this is only specified for this type of review documents
	public void LoadDoc(String filename) {
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			StringBuffer buffer = new StringBuffer(1024);
			String line;

			while ((line = reader.readLine()) != null) {
				buffer.append(line);
			}
			reader.close();
			
			//How to generalize it to several classes???? 
			if(filename.contains("pos")){
				//Collect the number of documents in one class.
				AnalyzeDoc(new _Doc(m_corpus.getSize(), buffer.toString(), 0));				
			}else if(filename.contains("neg")){
				AnalyzeDoc(new _Doc(m_corpus.getSize(), buffer.toString(), 1));
			}
		} catch(IOException e){
			System.err.format("[Error]Failed to open file %s!!", filename);
			e.printStackTrace();
		}
	}
	
	//Given a long string, return a set of sentences using .?! as delimiter
	// added by Md. Mustafizur Rahman for HTMM Topic Modelling 
	protected String[] findSentence(String source){
		String regexp = "[.?!]+"; 
	    String [] sentences;
	    sentences = source.split(regexp);
	    return sentences;
	}
	
	/*Analyze a document and add the analyzed document back to corpus.
	 *In the case CV is not loaded, we need two if loops to check.
	 * The first is if the term is in the vocabulary.***I forgot to check this one!
	 * The second is if the term is in the sparseVector.
	 * In the case CV is loaded, we still need two if loops to check.*/
	protected void AnalyzeDoc(_Doc doc) {
		String[] tokens = TokenizerNormalizeStemmer(doc.getSource());// Three-step analysis.
		//When we load the docuemnts for selecting CV, we ignore the documents.
		if (tokens.length< m_lengthThreshold) return;

		HashMap<Integer, Double> spVct = new HashMap<Integer, Double>(); // Collect the index and counts of features.
		int index = 0;
		double value = 0;
		// Construct the sparse vector.
		for (String token : tokens) {
			// CV is not loaded, take all the tokens as features.
			if (!m_isCVLoaded) {
				if (m_featureNameIndex.containsKey(token)) {
					index = m_featureNameIndex.get(token);
					if (spVct.containsKey(index)) {
						value = spVct.get(index) + 1;
						spVct.put(index, value);
					} else {
						spVct.put(index, 1.0);
						m_featureStat.get(token).addOneDF(doc.getYLabel());
					}
				} else {// indicate we allow the analyzer to dynamically expand the feature vocabulary
					expandVocabulary(token);// update the m_featureNames.
					index = m_featureNameIndex.get(token);
					spVct.put(index, 1.0);
					m_featureStat.get(token).addOneDF(doc.getYLabel());
				}
				m_featureStat.get(token).addOneTTF(doc.getYLabel());
			} else if (m_featureNameIndex.containsKey(token)) {// CV is loaded.
				index = m_featureNameIndex.get(token);
				if (spVct.containsKey(index)) {
					value = spVct.get(index) + 1;
					spVct.put(index, value);
				} else {
					spVct.put(index, 1.0);
					m_featureStat.get(token).addOneDF(doc.getYLabel());
				}
				m_featureStat.get(token).addOneTTF(doc.getYLabel());
			}
			// if the token is not in the vocabulary, nothing to do.
		}
		
		m_classMemberNo[doc.getYLabel()]++;
		
		if (m_isCVLoaded){
			if (spVct.size() >= 1) {//temporary code for debugging purpose
				doc.createSpVct(spVct);
				m_corpus.addDoc(doc);
//				if (m_releaseContent)
//					doc.clearSource();
				return;
			} else return;
		} else return;
	}
	
	protected void AnalyzeDocWithProjFeatures(_Doc doc){
	
		String[] tokens = TokenizerNormalizeStemmer(doc.getSource());// Three-step analysis.
		//When we load the docuemnts for selecting CV, we ignore the documents.
		if (tokens.length< m_lengthThreshold) return;

		int index = 0, projIndex = 0; 
		double value = 0, projValue = 0;
		HashMap<Integer, Double> spVct = new HashMap<Integer, Double>(); // Collect the index and counts of features.
		HashMap<Integer, Double> projectedVct = new HashMap<Integer, Double>();//Collect the index and counts of projected features.
		// Construct the sparse vector.
		for (String token : tokens) {
			if (m_featureNameIndex.containsKey(token)) {// Check if the word in the CV.
				index = m_featureNameIndex.get(token);
				if (spVct.containsKey(index)) {
					value = spVct.get(index) + 1;
					spVct.put(index, value);
				} else {
					spVct.put(index, 1.0);
					m_featureStat.get(token).addOneDF(doc.getYLabel());
				}
				m_featureStat.get(token).addOneTTF(doc.getYLabel());
			}
			if(m_projFeatureNameIndex.containsKey(token)){// Check if the word in the projected CV.
				projIndex = m_projFeatureNameIndex.get(token);
				if(projectedVct.containsKey(projIndex)){
					projValue = projectedVct.get(projIndex) + 1;
					projectedVct.put(projIndex, projValue);
				} else{
					projectedVct.put(projIndex, 1.0);
					//m_projFeatureStat.get(tmpToken).addOneTTF(doc.getYLabel());
				}
			} else{
				projIndex = m_projFeatureNameIndex.size();
				m_projFeatureNameIndex.put(token, projIndex);
				projectedVct.put(projIndex, 1.0);
			}
		}
		
		m_classMemberNo[doc.getYLabel()]++;
		if (spVct.size()>= 1) {//temporary code for debugging purpose
			doc.createSpVct(spVct);
			if(projectedVct.size() >=1 ) //If it has projected vector, create it.
				doc.createProjVct(projectedVct);
			m_corpus.addDoc(doc);
//			if (m_releaseContent)
//				doc.clearSource();
			return;
		} else return;
	}
	
	// adding sentence splitting function, modified for HTMM
	protected void AnalyzeDocWithStnSplit(_Doc doc) {
		String[] sentences = m_stnDetector.sentDetect(doc.getSource());
		HashMap<Integer, Double> spVct = new HashMap<Integer, Double>(); // Collect the index and counts of features.
		ArrayList<_SparseFeature[]> stnList = new ArrayList<_SparseFeature[]>(); // to avoid empty sentences
		
		for(String sentence : sentences) {
			String[] tokens = TokenizerNormalizeStemmer(sentence);// Three-step analysis.			
			int index = 0;
			double value = 0;
			HashMap<Integer, Double> sentence_vector = new HashMap<Integer, Double>(); 
			
			// Construct the sparse vector.
			for (String token : tokens) {
				// CV is not loaded, take all the tokens as features.
				if (!m_isCVLoaded) {
					if (m_featureNameIndex.containsKey(token)) {
						index = m_featureNameIndex.get(token);
						if (spVct.containsKey(index)) {
							value = spVct.get(index) + 1;
							spVct.put(index, value);
							if(sentence_vector.containsKey(index)){
								value = sentence_vector.get(index) + 1;
								sentence_vector.put(index, value);
							} else {
								sentence_vector.put(index, 1.0);
							}
												
						} else {
							spVct.put(index, 1.0);
							sentence_vector.put(index, 1.0);
							m_featureStat.get(token).addOneDF(doc.getYLabel());
						}
					} else {// indicate we allow the analyzer to dynamically expand the feature vocabulary
						expandVocabulary(token);// update the m_featureNames.
						index = m_featureNameIndex.get(token);
						spVct.put(index, 1.0);
						sentence_vector.put(index, 1.0);
				    	m_featureStat.get(token).addOneDF(doc.getYLabel());
					}
	
					m_featureStat.get(token).addOneTTF(doc.getYLabel());
				} else if (m_featureNameIndex.containsKey(token)) {// CV is loaded.
					index = m_featureNameIndex.get(token);
					if (spVct.containsKey(index)) {
						value = spVct.get(index) + 1;
						spVct.put(index, value);
						if(sentence_vector.containsKey(index)){
							value = sentence_vector.get(index) + 1;
							sentence_vector.put(index, value);
						} else {
							sentence_vector.put(index, 1.0);
						}
					} else {
						spVct.put(index, 1.0);
						sentence_vector.put(index, 1.0);
					
						m_featureStat.get(token).addOneDF(doc.getYLabel());
					}
					m_featureStat.get(token).addOneTTF(doc.getYLabel());
				}
			}// End for loop for token
			
			if (sentence_vector.size()>0)//avoid empty sentence
				stnList.add(Utils.createSpVct(sentence_vector));
		} // End For loop for sentence	
		
		m_classMemberNo[doc.getYLabel()]++;
		//the document should be long enough
		if (spVct.size()>=m_lengthThreshold && stnList.size()>1) { 
			doc.createSpVct(spVct);
			doc.setSentences(stnList);
			m_corpus.addDoc(doc);
			
			if (m_releaseContent)
				doc.clearSource();
			return;
		} else
			return;
	}
	
	//Analyze document with POS Tagging.
	protected void AnalyzeDocWithPOSTagging(_Doc doc) {
		if ((TokenizerNormalizeStemmer(doc.getSource()).length) < m_lengthThreshold) return;
		int index = 0, projIndex = 0; 
		double value = 0, projValue = 0;
		HashMap<Integer, Double> spVct = new HashMap<Integer, Double>(); // Collect the index and counts of features.
		HashMap<Integer, Double> projectedVct = new HashMap<Integer, Double>();//Collect the index and counts of projected features.
		int[] featureArray = new int[m_featureDimension]; //Array used to store feature statistics of docs.
		
		String[] sentences = m_stnDetector.sentDetect(doc.getSource());//Split sentences first.
		for(String s: sentences){
			String[] tokens = Tokenizer(s);
			String[] tags = m_tagger.tag(tokens);
			for(int i = 0; i < tokens.length; i++){
				String tmpToken = SnowballStemming(Normalize(tokens[i]));
				if (isLegit(tmpToken)){
					//If the token is in the CV, build the sparse hashmap first.
					if(m_featureNameIndex.containsKey(tmpToken)) {// CV is loaded.
						index = m_featureNameIndex.get(tmpToken);
						if (spVct.containsKey(index)) {
							value = spVct.get(index) + 1;
							spVct.put(index, value);
						} else {
							spVct.put(index, 1.0);
							m_featureStat.get(tmpToken).addOneDF(doc.getYLabel());
						}
						m_featureStat.get(tmpToken).addOneTTF(doc.getYLabel());
						
						//POS Tagging part: pos 1 and pos 3.
						if(m_posTaggingMethod == 1){
							if(tags[i].equals("RB")||tags[i].equals("RBR")||tags[i].equals("RBS")||tags[i].equals("JJ")||tags[i].equals("JJR")||tags[i].equals("JJS")){
								if(m_projFeatureNameIndex.containsKey(tmpToken)){
									projIndex = m_projFeatureNameIndex.get(tmpToken);
									if(projectedVct.containsKey(projIndex)){
										projValue = projectedVct.get(projIndex) + 1;
										projectedVct.put(projIndex, projValue);
									} else{
										projectedVct.put(projIndex, 1.0);
										//m_projFeatureStat.get(tmpToken).addOneTTF(doc.getYLabel());
									}
								} else{
									projIndex = m_projFeatureNameIndex.size();
									m_projFeatureNameIndex.put(tmpToken, projIndex);
									projectedVct.put(projIndex, 1.0);
								}
							}
						} else if(m_posTaggingMethod == 3 || m_posTaggingMethod == 4){
							if(tags[i].equals("RB")||tags[i].equals("RBR")||tags[i].equals("RBS")){
								tmpToken = tokens[i] + "#r";
							} else if (tags[i].equals("JJ")||tags[i].equals("JJR")||tags[i].equals("JJS")){
								tmpToken = tokens[i] + "#a";
							} else if (tags[i].equals("NN")||tags[i].equals("NNS")||tags[i].equals("NNP")||tags[i].equals("NNPS")){
								tmpToken = tokens[i] + "#n";
							} else if (tags[i].equals("VB")||tags[i].equals("VBD")||tags[i].equals("VBG")||tags[i].equals("VBN")||tags[i].equals("VBP")||tags[i].equals("VBZ")){
								tmpToken = tokens[i] + "#v";
							} 
							if(m_posTaggingMethod == 3){
								if(m_dictionary.contains(tmpToken)){
									if(m_projFeatureNameIndex.containsKey(tmpToken)){
										projIndex = m_projFeatureNameIndex.get(tmpToken);
										if(projectedVct.containsKey(projIndex)){
											projValue = projectedVct.get(projIndex) + 1;
											projectedVct.put(projIndex, projValue);
										} else
											projectedVct.put(projIndex, 1.0);
									}
								} else {
									projIndex = m_projFeatureNameIndex.size();
									m_projFeatureNameIndex.put(tmpToken, projIndex);
									projectedVct.put(projIndex, 1.0);
								}
							} 
//							else if(m_posTaggingMethod == 4){
//								if(m_dictMap.containsKey(tmpToken)){
//									double score = m_dictMap.get(tmpToken);
////									if(score != 0){
//										m_projFeatureScore.put(tmpToken, score);
//										int ind = findIndex(score);
//										featureArray[ind]++;
////									}
//								}
//							}
							else if(m_posTaggingMethod == 4){
								if(m_sentiIndex.containsKey(tmpToken))
									featureArray[m_sentiIndex.get(tmpToken)]++;
							}
						}
					}
				}
			}
		}
		m_classMemberNo[doc.getYLabel()]++;
		if (spVct.size()>= 1) {//temporary code for debugging purpose
			doc.createSpVct(spVct);
			if(m_posTaggingMethod == 3 || m_posTaggingMethod ==1)
				doc.createProjVct(projectedVct);
			else if(m_posTaggingMethod == 4)
				doc.createProjVct2(featureArray);
			m_corpus.addDoc(doc);
//			if (m_releaseContent)
//				doc.clearSource();
			return;
		} else return;
	}
	
	//Put the features into the groups according to their scores.
	public int findIndex(double score){
		if(score == -1) 
			return 0;
		int index = 0, start = 0, end = m_featureDimension;
		double[] values = new double[m_featureDimension+1];
		for(int i = 0; i < m_featureDimension + 1; i++){
			values[i] = 2.0 / m_featureDimension * i - 1.0;
		}
		for(int i = 0; i < values.length; i++){
			index = (start + end) / 2;
			if(values[index] < score)
				if(values[index+1] > score) break;
				else start = index;
			else if(values[index] >= score)
				if(values[index-1] < score){
					index = index-1;
					break;
				}
				else end = index;
			else break;
		}
		return index;
	}

	//Build the filter based on the raw filter and features.
	public int builderFilter(){
		for(String f: m_rawFilter){
			if(m_featureNameIndex.containsKey(f))
				m_filter.put(m_featureNameIndex.get(f), f);
		}
		return m_filter.size();
	}
	//Bulid the project vector for every document.
	public void buildProjectSpVct(){
		for(_Doc d: m_corpus.getCollection())
			d.setProjectedFv(m_filter);
	}
	
	//Save projected features to file.
	public void saveProjectedFvs(String projectedFeatureLocation) throws FileNotFoundException{
		if (projectedFeatureLocation==null || projectedFeatureLocation.isEmpty())
			return;
		
		System.out.format("Saving projected features to %s...\n", projectedFeatureLocation);
		PrintWriter writer = new PrintWriter(new File(projectedFeatureLocation));
		Collection<String> projectedFvs = m_filter.values();
		for (String s: projectedFvs) //printed out all the projected features.
			writer.println(s);
		writer.close();
	}
	
	//Load the sentinet word and store them in the dictionary for later use.
	public void LoadSNW(String filename) throws IOException {
		m_dictionary = new HashSet<String>();//for tagging3 to store features.
		// From String to list of doubles.
		HashMap<String, HashMap<Integer, Double>> tempDictionary = new HashMap<String, HashMap<Integer, Double>>();

		BufferedReader csv = null;
		try {
			csv = new BufferedReader(new FileReader(filename));
			int lineNumber = 0;
			String line;
			while ((line = csv.readLine()) != null) {
				lineNumber++;
				// If it's a comment, skip this line.
				if (!line.trim().startsWith("#")) {
					// We use tab separation
					String[] data = line.split("\t");
					String wordTypeMarker = data[0];
					// Is it a valid line? Otherwise, through exception.
					if (data.length != 6)
						throw new IllegalArgumentException("Incorrect tabulation format in file, line: " + lineNumber);
					// Calculate synset score as score = PosS - NegS. If it's 0, then it is neutral word, ignore it.
					Double synsetScore = Double.parseDouble(data[2])- Double.parseDouble(data[3]);
					// Get all Synset terms
					String[] synTermsSplit = data[4].split(" ");
					// Go through all terms of current synset.
					for (String synTermSplit : synTermsSplit) {
						// Get synterm and synterm rank
						String[] synTermAndRank = synTermSplit.split("#"); // able#1 = [able, 1]
						String synTerm = synTermAndRank[0] + "#" + wordTypeMarker; // able#a
						int synTermRank = Integer.parseInt(synTermAndRank[1]); // different senses of a word
						// Add the current term to map if it doesn't have one
						if (!tempDictionary.containsKey(synTerm))
							tempDictionary.put(synTerm, new HashMap<Integer, Double>());// <able#a, <<1, score>, <2, score>...>>
						// If the dict already has the synTerm, just add synset link-<2, score> to synterm.
						tempDictionary.get(synTerm).put(synTermRank, synsetScore);
					}
				}
			}
			// Go through all the terms.
			Set<String> synTerms = tempDictionary.keySet();

			for (String synTerm : synTerms) {
				double score = 0;
				HashMap<Integer, Double> synSetScoreMap = tempDictionary.get(synTerm);
				Collection<Double> scores = synSetScoreMap.values();
				for (double s : scores)
					score += s;
				if (score != 0){
					String[] termMarker = synTerm.split("#");
					m_dictionary.add(SnowballStemming(Normalize(termMarker[0])) + "#" + termMarker[1]);
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			if (csv != null) {
				csv.close();
			}
		}
	}
	//Load the sentinet word and store them in the dictionary for later use.
	public void LoadSNWWithScore(String filename) throws IOException {
		m_dictMap = new HashMap<String, Double>();//for tagging3 to store features.
		// From String to list of doubles.
		HashMap<String, HashMap<Integer, Double>> tempDictionary = new HashMap<String, HashMap<Integer, Double>>();

		BufferedReader csv = null;
		try { 
			csv = new BufferedReader(new FileReader(filename));
			int lineNumber = 0;
			String line;
			while ((line = csv.readLine()) != null) {
				lineNumber++;
				// If it's a comment, skip this line.
				if (!line.trim().startsWith("#")) {
					// We use tab separation
					String[] data = line.split("\t");
					String wordTypeMarker = data[0];
					// Is it a valid line? Otherwise, through exception.
					if (data.length != 6)
						throw new IllegalArgumentException("Incorrect tabulation format in file, line: " + lineNumber);
					// Calculate synset score as score = PosS - NegS. If it's 0, then it is neutral word, ignore it.
					Double synsetScore = Double.parseDouble(data[2]) - Double.parseDouble(data[3]);
					// Get all Synset terms
					String[] synTermsSplit = data[4].split(" ");
					// Go through all terms of current synset.
					for (String synTermSplit : synTermsSplit) {
						// Get synterm and synterm rank
						String[] synTermAndRank = synTermSplit.split("#"); // able#1 = [able, 1]
						String synTerm = synTermAndRank[0] + "#" + wordTypeMarker; // able#a
						int synTermRank = Integer.parseInt(synTermAndRank[1]); // different senses of a word
						// Add the current term to map if it doesn't have one
						if (!tempDictionary.containsKey(synTerm))
							tempDictionary.put(synTerm, new HashMap<Integer, Double>());// <able#a, <<1, score>, <2, score>...>>
						// If the dict already has the synTerm, just add synset link-<2, score> to synterm.
						tempDictionary.get(synTerm).put(synTermRank, synsetScore);
					}
				}
			}

			// Go through all the terms.
			Set<String> synTerms = tempDictionary.keySet();
			for (String synTerm : synTerms) {
				double score = 0;
				int count = 0;
				HashMap<Integer, Double> synSetScoreMap = tempDictionary.get(synTerm);
				Collection<Double> scores = synSetScoreMap.values();
				for (double s : scores){
					if(s != 0){
						score += s;
						count++;
					}
					if(score != 0)
						score = (double) score / count;
				}
				String[] termMarker = synTerm.split("#");
				m_dictMap.put(SnowballStemming(Normalize(termMarker[0])) + "#" + termMarker[1], score);
			}
			
		} finally {
			if (csv != null) {
				csv.close();
			}
		}
	}
	
	//Sort the m_dictMap by values.
	public void AssignFeatureIndexes(){
		//Sort projected features according to their values.
		ArrayList<String> keys = new ArrayList<String>();
		keys.addAll(m_projFeatureScore.keySet());
		Collections.sort(keys, new Comparator<String>() {
			public int compare(String s1, String s2){
				double d1 = m_projFeatureScore.get(s1);
				double d2 = m_projFeatureScore.get(s2);
				if(d1 > d2)
					return -1;
				else if(d1 < d2)
					return 1;
				else return 0;
			}
		});
		
		int dictSize = m_projFeatureScore.size(), interval = 0;
		m_sentiIndex = new HashMap<String, Integer>();
	
		if (dictSize % m_featureDimension == 0)
			interval = m_projFeatureScore.size() / m_featureDimension;
		else 
			interval = m_projFeatureScore.size() / m_featureDimension + 1;
		
		for(int i = 0; i < m_featureDimension; i++ ){
			for(int j = 0; j < interval; j++){
				int index = i * interval + j;
				if(index < dictSize){
					String word = keys.get(index);
					m_sentiIndex.put(word, m_featureDimension -1 -i);
//					System.out.print(String.format("%s, %.3f, %d\n", word, m_projFeatureScore.get(word), (m_featureDimension -1 -i)));
				}
			}
		}
	}
	//General methods of sorting hashMap by values.
	public HashMap<String, Double> SortHashMapByValues(final HashMap<String, Double> map){
		ArrayList<String> keys = new ArrayList<String>();
		HashMap<String, Double> sortedMap = new HashMap<String, Double>();
		keys.addAll(map.keySet());
		Collections.sort(keys, new Comparator<String>() {
			public int compare(String s1, String s2){
				double d1 = map.get(s1);
				double d2 = map.get(s2);
				if(d1 > d2)
					return 1;
				else if(d1 < d2)
					return -1;
				else return 0;
			}
		});
		for(String s: keys){
			sortedMap.put(s, map.get(s));
			System.out.println(s + "\t" + map.get(s));
		}
		return sortedMap;
	}
//	public static void main(String[] args){
//		HashMap<String, Double> test = new HashMap<String, Double>();
//		test.put("a", 1.0);
//		test.put("b", 1.0);
//		test.put("c", 0.1);
//		test.put("d", 0.5);
//		test.put("mm", -1.0);
//		sortHashMap(test);
//	}

	public void setFeatureDimension(int k){
		m_featureDimension = k;
	}
	
	//Save all the words in sentiword net.
	public void saveSentiWordNetFeatures(String filename) throws FileNotFoundException{
		PrintWriter writer = new PrintWriter(new File(filename));
		for(String f: m_dictMap.keySet()){
			writer.format("%s,%.3f\n", f, m_dictMap.get(f));
		}
		writer.close();
	}
	
	//Save all the words that both in sentiwordnet and in general features.
	public void saveProjFeaturesScores(String filename) throws FileNotFoundException{
		PrintWriter writer = new PrintWriter(new File(filename));
		for(String s: m_projFeatureScore.keySet())
			writer.format("%s,%.3f\n", s, m_projFeatureScore.get(s));
		System.out.println(m_projFeatureScore.size() + " proj features are saves.");
		writer.close();
	}
	
	public void resetStopwords(){
		m_stopwords.clear();
	}
}	

