package Classifier.semisupervised;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;

import structures._Corpus;
import structures._Doc;
import structures._RankItem;
import structures._SparseFeature;

public class GaussianFieldsByRandomWalk extends GaussianFields {
	double m_difference; //The difference between the previous labels and current labels.
	double m_eta; //The parameter used in random walk. 
	double[] m_fu_last; // result from last round of random walk
	
	double m_delta; // convergence criterion for random walk
	boolean m_storeGraph; // shall we precompute and store the graph
//	int count = 0;
	//Default constructor without any default parameters.
	public GaussianFieldsByRandomWalk(_Corpus c, int classNumber, int featureSize, String classifier){
		super(c, classNumber, featureSize, classifier);
		
		m_eta = 0.1;
		m_labelRatio = 0.1;
		m_delta = 1e-5;
		m_storeGraph = false;
	}	
	
	//Constructor: given k and kPrime
	public GaussianFieldsByRandomWalk(_Corpus c, int classNumber, int featureSize, String classifier, double ratio, int k, int kPrime, double alhpa, double beta, double delta, double eta, boolean storeGraph){
		super(c, classNumber, featureSize, classifier, ratio, k, kPrime);
		
		m_alpha = alhpa;
		m_beta = beta;
		m_delta = delta;
		m_eta = eta;
		m_storeGraph = storeGraph;
	}

	@Override
	public String toString() {
		return String.format("Gaussian Fields by random walk [C:%s, k:%d, k':%d, r:%.3f, alpha:%.3f, beta:%.3f, eta:%.3f]", m_classifier, m_k, m_kPrime, m_labelRatio, m_alpha, m_beta, m_eta);
	}
	
	//The random walk algorithm to generate new labels for unlabeled data.
	//Take the average of all neighbors as the new label until they converge.
	void randomWalk(){//construct the sparse graph on the fly every time
		double wL = m_alpha / (m_k + m_beta*m_kPrime), wU = m_beta * wL;
		
		/**** Construct the C+scale*\Delta matrix and Y vector. ****/
		for (int i = 0; i < m_U; i++) {
			double wijSumU = 0, wijSumL = 0;
			double fSumU = 0, fSumL = 0;
			
			/****Construct the top k' unlabeled data for the current data.****/
			for (int j = 0; j < m_U; j++) {
				if (j == i)
					continue;
				m_kUU.add(new _RankItem(j, getCache(i, j)));
			}
			/****Construct the top k labeled data for the current data.****/
			for (int j = 0; j < m_L; j++)
				m_kUL.add(new _RankItem(m_U + j, getCache(i, m_U + j)));
			
			/****Get the sum of k'UU******/
			for(_RankItem n: m_kUU){
				wijSumU += n.m_value; //get the similarity between two nodes.
//				fSumU += n.m_value * m_fu_last[n.m_index];
				fSumU += n.m_value * m_fu[n.m_index];
			}
			m_kUU.clear();
			
			/****Get the sum of kUL******/
			for(_RankItem n: m_kUL){
				wijSumL += n.m_value;
				fSumL += n.m_value * m_Y[n.m_index];
			}
			m_kUL.clear();
			
			if(wijSumL!=0 || wijSumU!=0){
				m_fu[i] = m_eta * (fSumL*wL + fSumU*wU) / (wijSumL*wL + wijSumU*wU) + (1-m_eta) * m_Y[i];
			}
			if(Double.isNaN(m_fu[i]))
				System.out.println("NaN detected!!!");
		}
	}
	
	//based on the precomputed sparse graph
	void randomWalkWithGraph(){
		double wij, wL = m_alpha / (m_k + m_beta*m_kPrime), wU = m_beta * wL;
		
		/**** Construct the C+scale*\Delta matrix and Y vector. ****/
		for (int i = 0; i < m_U; i++) {
			double wijSumU = 0, wijSumL = 0;
			double fSumU = 0, fSumL = 0;
			int j = 0;
			
			/****Get the sum of k'UU******/
			for (; j < m_U; j++) {
				if (j == i) 
					continue;
				wij = m_graph.getQuick(i, j); //get the similarity between two nodes.
				if (wij == 0)
					continue;
				
				wijSumU += wij;
				//fSumU += wij * m_fu_last[j];//use the old results
				fSumU += wij * m_fu[j];//use the updated results immediately
			}
			
			/****Get the sum of kUL******/
			for (; j<m_U+m_L; j++) {
				wij = m_graph.getQuick(i, j); //get the similarity between two nodes.
				if (wij == 0)
					continue;
				
				wijSumL += wij;
				fSumL += wij * m_Y[j];
			}
			
			if(wijSumL!=0 || wijSumU!=0)
				m_fu[i] = m_eta * (fSumL*wL + fSumU*wU) / (wijSumL*wL + wijSumU*wU) + (1-m_eta) * m_Y[i];

			if(Double.isNaN(m_fu[i]))
				System.out.println("NaN detected!!!");
		}
	}
	
	double updateFu() {
		m_difference = 0;
		for(int i = 0; i < m_U; i++){
			m_difference += Math.abs(m_fu[i] - m_fu_last[i]);
			m_fu_last[i] = m_fu[i];//record the last result
		}
		return m_difference/m_U;
	}
	
	//The test for random walk algorithm.
	public double test(){
		/***Construct the nearest neighbor graph****/
		constructGraph(m_storeGraph);
		
		if (m_fu_last==null || m_fu_last.length<m_U)
			m_fu_last = new double[m_U]; //otherwise we can reuse the current memory
		
		//initialize fu and fu_last
		for(int i=0; i<m_U; i++) {
			m_fu[i] = m_Y[i];
			m_fu_last[i] = m_Y[i];//random walk starts from multiple learner
		}
		
		/***use random walk to solve matrix inverse***/
		do {
			if (m_storeGraph)
				randomWalkWithGraph();
			else
				randomWalk();			
		} while(updateFu() > m_delta);
		
		/***get some statistics***/
		for(int i = 0; i < m_U; i++){
			for(int j=0; j<m_classNo; j++)
				m_pYSum[j] += Math.exp(-Math.abs(j-m_fu[i]));			
		}
		
		/***evaluate the performance***/
		double acc = 0;
		int pred, ans;
		try{
			for(int i = 0; i < m_U; i++) {
				pred = getLabel(m_fu[i]);
				ans = m_testSet.get(i).getYLabel();
				m_TPTable[pred][ans] += 1;
			
				int SVMpred = m_classifier.predict(m_testSet.get(i));
				if(!(SVMpred == pred && pred == ans)){
					m_writerFuSVM.write(String.format("%.3f, %d, %d, %d\n", m_fu[i], SVMpred, pred, ans));
				}
				//SVM correct, RW incorrect.
				if(SVMpred == ans && pred != ans){
					debugWrongRW(m_testSet.get(i), true);
				}	
				//SVM incorrect, RW correct.
				if(SVMpred != ans && pred == ans){
					debugWrongRW(m_testSet.get(i), false);
				}	
				if (pred != ans) {
					m_count++;
					if (m_debugOutput!=null){
						if (m_POSTagging != 0)
							debugWithPOSTagging(m_testSet.get(i));
						else 
							m_debugWriter.write(m_count +"\n");
							debug(m_testSet.get(i));
					}
				} else 
					acc ++;
			}
		} catch (IOException e){
			e.printStackTrace();
		}
		
//		System.out.println("count is " + count);
		m_precisionsRecalls.add(calculatePreRec(m_TPTable));
		return acc/m_U;
	}
	
	public void debugWrongRW(_Doc d, boolean flag) throws IOException{
		BufferedWriter writer;
		if(flag)
			writer = m_writerWrongRW;
		else 
			writer = m_writerWrongSVM;
		
		int id = d.getID();
		_SparseFeature[] dsfs = d.getSparse();
		_RankItem item;
		_Doc neighbor;
		double sim, wijSumU=0, wijSumL=0;
		
		writer.write("============================================================================\n");
		writer.write(String.format("Label:%d, fu:%.4f, getLabel1:%d, getLabel3:%d, SVM:%d, Content:%s\n", d.getYLabel(), m_fu[id], getLabel(m_fu[id]), getLabel3(m_fu[id]), (int)m_Y[id], d.getSource()));
		
		for(int i = 0; i< dsfs.length; i++){
			String feature = m_IndexFeature.get(dsfs[i].getIndex());
			writer.write(String.format("(%s %.4f),", feature, dsfs[i].getValue()));
		}
		writer.write("\n");
		
		//find top five labeled
		/****Construct the top k labeled data for the current data.****/
		for (int j = 0; j < m_L; j++)
			m_kUL.add(new _RankItem(j, getCache(id, m_U + j)));
		
		/****Get the sum of kUL******/
		for(_RankItem n: m_kUL)
			wijSumL += n.m_value; //get the similarity between two nodes.
		
		/****Get the top 5 elements from kUL******/
		writer.write("*************************Labeled data*************************************\n");
		for(int k=0; k < 5; k++){
			item = m_kUL.get(k);
			neighbor = m_labeled.get(item.m_index);
			sim = item.m_value/wijSumL;
			
			//Print out the sparse vectors of the neighbors.
			writer.write(String.format("Label:%d, Similarity:%.4f\n", neighbor.getYLabel(), sim));
//			writer.write(neighbor.getSource()+"\n");
			_SparseFeature[] sfs = neighbor.getSparse();
			int pointer1 = 0, pointer2 = 0;
			//Find out all the overlapping features and print them out.
			while(pointer1 < dsfs.length && pointer2 < sfs.length){
				_SparseFeature tmp1 = dsfs[pointer1];
				_SparseFeature tmp2 = sfs[pointer2];
				if(tmp1.getIndex() == tmp2.getIndex()){
					String feature = m_IndexFeature.get(tmp1.getIndex());
					writer.write(String.format("(%s %.4f),", feature, tmp2.getValue()));
					pointer1++;
					pointer2++;
				} else if(tmp1.getIndex() < tmp2.getIndex())
					pointer1++;
				else pointer2++;
			}
			writer.write("\n");
		}
		m_kUL.clear();
		
		//find top five unlabeled
		/****Construct the top k' unlabeled data for the current data.****/
		for (int j = 0; j < m_U; j++) {
			if (j == id)
				continue;
			m_kUU.add(new _RankItem(j, getCache(id, j)));
		}
		
		/****Get the sum of k'UU******/
		for(_RankItem n: m_kUU)
			wijSumU += n.m_value; //get the similarity between two nodes.
		
		/****Get the top 5 elements from k'UU******/
		writer.write("*************************Unlabeled data*************************************\n");
		for(int k=0; k<5; k++){
			item = m_kUU.get(k);
			neighbor = m_testSet.get(item.m_index);
			sim = item.m_value/wijSumU;
			
			writer.write(String.format("True Label:%d, f_u:%.4f, Similarity:%.4f\n", neighbor.getYLabel(), m_fu[neighbor.getID()], sim));
//			writer.write(neighbor.getSource()+"\n");
			_SparseFeature[] sfs = neighbor.getSparse();
			int pointer1 = 0, pointer2 = 0;
			//Find out all the overlapping features and print them out.
			while(pointer1 < dsfs.length && pointer2 < sfs.length){
				_SparseFeature tmp1 = dsfs[pointer1];
				_SparseFeature tmp2 = sfs[pointer2];
				if(tmp1.getIndex() == tmp2.getIndex()){
					String feature = m_IndexFeature.get(tmp1.getIndex());
					writer.write(String.format("(%s %.4f),", feature, tmp2.getValue()));
					pointer1++;
					pointer2++;
				} else if(tmp1.getIndex() < tmp2.getIndex())
					pointer1++;
				else pointer2++;
			}
			writer.write("\n");
		}
		m_kUU.clear();
	}
	
	//If we use pos tagging, then projected vectors are used to calculate similarity.
	protected void debugWithPOSTagging(_Doc d){
		int id = d.getID();
		_SparseFeature[] dsfs = d.getProjectedFv();
		_RankItem item;
		_Doc neighbor;
		double sim, wijSumU=0, wijSumL=0;
		
		try {
			m_debugWriter.write("============================================================================\n");
			m_debugWriter.write(String.format("Label:%d, fu:%.4f, getLabel1:%d, getLabel3:%d, SVM:%d, Content:%s\n", d.getYLabel(), m_fu[id], getLabel(m_fu[id]), getLabel3(m_fu[id]), (int)m_Y[id], d.getSource()));
			
			for(int i = 0; i< dsfs.length; i++){
				if( m_POSTagging !=4 ){
					String feature = m_IndexFeature.get(dsfs[i].getIndex());
					m_debugWriter.write(String.format("(%s %.4f),", feature, dsfs[i].getValue()));
				} else
					m_debugWriter.write(String.format("(%d %.2f),", dsfs[i].getIndex(), dsfs[i].getValue()));
			}
			m_debugWriter.write("\n");
			
			//find top five labeled
			/****Construct the top k labeled data for the current data.****/
			for (int j = 0; j < m_L; j++)
				m_kUL.add(new _RankItem(j, getCache(id, m_U + j)));
			
			/****Get the sum of kUL******/
			for(_RankItem n: m_kUL)
				wijSumL += n.m_value; //get the similarity between two nodes.
			
			/****Get the top 5 elements from kUL******/
			m_debugWriter.write("*************************Labeled data*************************************\n");
			for(int k=0; k < 10; k++){
				item = m_kUL.get(k);
				neighbor = m_labeled.get(item.m_index);
				sim = item.m_value/wijSumL;
				
				//Print out the sparse vectors of the neighbors.
				m_debugWriter.write(String.format("Label:%d, Similarity:%.4f\n", neighbor.getYLabel(), sim));
				_SparseFeature[] sfs = neighbor.getProjectedFv();
				int pointer1 = 0, pointer2 = 0;
				//Find out all the overlapping features and print them out.
				while(pointer1 < dsfs.length && pointer2 < sfs.length){
					_SparseFeature tmp1 = dsfs[pointer1];
					_SparseFeature tmp2 = sfs[pointer2];
					if(tmp1.getIndex() == tmp2.getIndex()){
						if( m_POSTagging !=4 ){
							String feature = m_IndexFeature.get(tmp2.getIndex());
							m_debugWriter.write(String.format("(%s %.4f),", feature, tmp2.getValue()));
						} else
							m_debugWriter.write(String.format("(%d %.2f),", tmp2.getIndex(), tmp2.getValue()));
						
						pointer1++;
						pointer2++;
					} else if(tmp1.getIndex() < tmp2.getIndex())
						pointer1++;
					else pointer2++;
				}
				m_debugWriter.write("\n");
			}
			m_kUL.clear();
			
			//find top five unlabeled
			/****Construct the top k' unlabeled data for the current data.****/
			for (int j = 0; j < m_U; j++) {
				if (j == id)
					continue;
				m_kUU.add(new _RankItem(j, getCache(id, j)));
			}
			
			/****Get the sum of k'UU******/
			for(_RankItem n: m_kUU)
				wijSumU += n.m_value; //get the similarity between two nodes.
			
			/****Get the top 5 elements from k'UU******/
			m_debugWriter.write("*************************Unlabeled data*************************************\n");
			for(int k=0; k<10; k++){
				item = m_kUU.get(k);
				neighbor = m_testSet.get(item.m_index);
				sim = item.m_value/wijSumU;
				
				m_debugWriter.write(String.format("True Label:%d, f_u:%.4f, Similarity:%.4f\n", neighbor.getYLabel(), m_fu[neighbor.getID()], sim));
				_SparseFeature[] sfs = neighbor.getProjectedFv();
				int pointer1 = 0, pointer2 = 0;
				//Find out all the overlapping features and print them out.
				while(pointer1 < dsfs.length && pointer2 < sfs.length){
					_SparseFeature tmp1 = dsfs[pointer1];
					_SparseFeature tmp2 = sfs[pointer2];
					if(tmp1.getIndex() == tmp2.getIndex()){
						if( m_POSTagging !=4 ){
							String feature = m_IndexFeature.get(tmp2.getIndex());
							m_debugWriter.write(String.format("(%s %.4f),", feature, tmp2.getValue()));
						} else
							m_debugWriter.write(String.format("(%d %.2f),", tmp2.getIndex(), tmp2.getValue()));
						
						pointer1++;
						pointer2++;
					} else if(tmp1.getIndex() < tmp2.getIndex())
						pointer1++;
					else pointer2++;
				}
				m_debugWriter.write("\n");
			}
			m_kUU.clear();
		} catch (IOException e) {
			e.printStackTrace();
		}
	} 
}
