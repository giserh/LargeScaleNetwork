/**
 * 
 */
package Analyzer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;

import json.JSONArray;
import json.JSONException;
import json.JSONObject;
import opennlp.tools.util.InvalidFormatException;
import structures.Post;
import structures.Product;
import structures._Doc;
import structures._stat;
import utils.Utils;

/**
 * @author hongning
 * Sample codes for demonstrating OpenNLP package usage 
 */
public class jsonAnalyzer extends DocAnalyzer{
	
	private SimpleDateFormat m_dateFormatter;
//	private ArrayList<Product> m_products; // All the products in one category.
//	private ArrayList<Integer> m_NumOfReviews; //The number of reviews of a product
//	private ArrayList<Double> m_Ratings; //The value of ratings

	
	//Constructor with ngram and fValue.
	public jsonAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold) throws InvalidFormatException, FileNotFoundException, IOException {
		super(tokenModel, classNo, providedCV, Ngram, threshold);
		m_dateFormatter = new SimpleDateFormat("MMMMM dd,yyyy");// standard date format for this project
//		m_products = new ArrayList<Product>();

	}
	
	//Constructor with ngram and fValue.
	public jsonAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold, String stnModel) throws InvalidFormatException, FileNotFoundException, IOException {
		super(tokenModel, stnModel, classNo, providedCV, Ngram, threshold);
		m_dateFormatter = new SimpleDateFormat("MMMMM dd,yyyy");// standard date format for this project
//		m_products = new ArrayList<Product>();
	}
	
	//Load a document and analyze it.
	@Override
	public void LoadDoc(String filename) {
		Product prod = null;
		JSONArray jarray = null;
		
		try {
			JSONObject json = LoadJson(filename);
			prod = new Product(json.getJSONObject("ProductInfo"));
			jarray = json.getJSONArray("Reviews");
		} catch (Exception e) {
			System.out.print('X');
			return;
		}	
//		double rating = 0;
//		double reviews = 0;
		for(int i=0; i<jarray.length(); i++) {
			try {
				Post post = new Post(jarray.getJSONObject(i));
				if (checkPostFormat(post)){
					long timeStamp = m_dateFormatter.parse(post.getDate()).getTime();
					String content;
					if (Utils.endWithPunct(post.getTitle()))
						content = post.getTitle() + " " + post.getContent();
					else
						content = post.getTitle() + ". " + post.getContent();
					_Doc review = new _Doc(m_corpus.getSize(), post.getID(), content, prod.getID(), post.getLabel()-1, timeStamp);
//					rating += review.getYLabel();
//					reviews++;
					if(this.m_stnDetector!=null)
						AnalyzeDocWithStnSplit(review);
					else
						AnalyzeDoc(review);
				}
			} catch (ParseException e) {
				System.out.print('T');
			} catch (JSONException e) {
				System.out.print('P');
			}
		}
//		rating = rating / reviews;
//		prod.setNumOfReviews((int)reviews);
//		prod.setRating(rating);
//		rating = 0;
//		reviews = 0;
//		m_products.add(prod);
	}
	
	//sample code for loading the json file
	public JSONObject LoadJson(String filename) {
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			StringBuffer buffer = new StringBuffer(1024);
			String line;
			
			while((line=reader.readLine())!=null) {
				//System.out.println(line);
				buffer.append(line);
			}
			reader.close();
			return new JSONObject(buffer.toString());
		} catch (Exception e) {
			System.out.print('X');
			return null;
		}
	}
	
	//check format for each post
	private boolean checkPostFormat(Post p) {
		if (p.getLabel() <= 0 || p.getLabel() > 5){
			//System.err.format("[Error]Missing Lable or wrong label!!");
			System.out.print('L');
			return false;
		}
		else if (p.getContent() == null){
			//System.err.format("[Error]Missing content!!\n");
			System.out.print('C');
			return false;
		}	
		else if (p.getDate() == null){
			//System.err.format("[Error]Missing date!!\n");
			System.out.print('d');
			return false;
		}
		else {
			// to check if the date format is correct
			try {
				m_dateFormatter.parse(p.getDate());
				//System.out.println(p.getDate());
				return true;
			} catch (ParseException e) {
				System.out.print('D');
			}
			return true;
		} 
	}
	
	//Save the reviews and rating in ./data/ReviewRating.xlsx
//	public void saveReviewRating(String path) throws FileNotFoundException{
//		//Sort the products according to the number of reviews.
//		Collections.sort(m_products, new Comparator<Product>(){
//			public int compare(Product p1, Product p2){
//				if (p1.getNumOfReviews() > p2.getNumOfReviews()) return 1;
//				else if (p1.getNumOfReviews() < p2.getNumOfReviews())return -1;
//				else return 0;
//			}
//		});
//		
//		if (path == null || path.isEmpty())
//			return;
//		
//		PrintWriter writer = new PrintWriter(new File(path));
//		for(Product p: m_products){
//			double showRating = p.getRating() + 1;
//			showRating = showRating / 5 * 2510;
//			writer.print(p.getID() + "\t" + p.getNumOfReviews() + "\t" + showRating + "\n");
//		}
//		writer.close();
//	}
}
