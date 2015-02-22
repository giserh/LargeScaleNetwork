package prapplication;

import java.util.Random;

import optimization.gradientBasedMethods.ProjectedGradientDescent;
import optimization.gradientBasedMethods.stats.OptimizerStats;
import optimization.linesearch.ArmijoLineSearchMinimizationAlongProjectionArc;
import optimization.linesearch.InterpolationPickFirstStep;
import optimization.linesearch.LineSearchMethod;
import optimization.stopCriteria.CompositeStopingCriteria;
import optimization.stopCriteria.ProjectedGradientL2Norm;
import optimization.stopCriteria.StopingCriteria;

public class coinTossPrOpt {

	double theta_a = 0.45;
	double theta_b = 0.45;
	
	double ratio_a;
	double ratio_b;
	
	double theta_a_true = 0.8, theta_b_true = 0.3;
	
	double count[][]; // 2 coins and 2 possible values
	int sample[][];
	
    public coinTossPrOpt() {
		initialize();
	}
	

	private int getBinomial(int n, double p) {
		  int x = 0;
		  for(int i = 0; i < n; i++) {
		    if(Math.random() < p)
		      x++;
		  }
		  return x;
	}
	
	public void em() {		
		clear_all();
		double theta_a_old = this.theta_a;
		double theta_b_old = this.theta_b;
		for(int i=0; i<100; i++){
			E_step();
			M_step();
			clear_all();
			System.out.println("Step: "+i+" Theta_a: "+theta_a+"; Theta_b: "+theta_b);
			
			double delta_a = Math.abs(theta_a - theta_a_old);//why did you only check the convergence of theta_a???
			double delta_b = Math.abs(theta_b - theta_b_old);//why did you only check the convergence of theta_a???
			
			if(delta_a<1e-12 && delta_b<1e-12)
				break;
			theta_a_old = theta_a;
			theta_b_old = theta_b;
		}		
	}

	private void clear_all() {
		// clear the fractional count array
		for(int i=0; i<2; i++){
			for(int j=0; j<2; j++){
				count[i][j] = 0;
			}
		}
	}
	
	private void M_step() {
		theta_a = count[0][0] / (count[0][0] + count[0][1]);
		theta_b = count[1][0] / (count[1][0] + count[1][1]);
	}

	
	public void initialize() {
		count = new double[2][2];
		sample = new int[5000][2]; // 5000 samples
	
		for(int i=0; i<sample.length; i++){
			Random cn = new Random();
			int coint_type = cn.nextInt(2); 
			
			if(coint_type == 1) {
				sample[i][0] = getBinomial(10, theta_a_true); // 10 number of tosses
				sample[i][1] = 10 - sample[i][0];
			}
			else{
				sample[i][0] = getBinomial(10, theta_b_true);  // 10 number of tosses
				sample[i][1] = 10 - sample[i][0];
			}
			
		}
	}
	
	private void E_step() {
		double gdelta = 1e-5, istp = 1.0;
		int maxStep = 50;
		for(int i=0; i<sample.length; i++){
			//data likelihood
			//IMPORTANT: you have assumed the probability of choosing one coin is 0.5!!!
			ratio_a = Math.pow(theta_a, sample[i][0]) * Math.pow(1.0-theta_a, sample[i][1]);
			ratio_b = Math.pow(theta_b, sample[i][0]) * Math.pow(1.0-theta_b, sample[i][1]);
			
			double sum = ratio_a + ratio_b;
			ratio_a /= sum;
			ratio_b /= sum;
			
			if (sample[i][0] != sample[i][1]) {
				//perform posterior regularization				
				problem_coin testcase = new problem_coin(ratio_a, sample[i][0] - sample[i][1]);			
				testcase.setDebugLevel(-1);
				
				LineSearchMethod ls = new ArmijoLineSearchMinimizationAlongProjectionArc(new InterpolationPickFirstStep(istp));
				ProjectedGradientDescent optimizer = new ProjectedGradientDescent(ls);
				StopingCriteria stopGrad = new ProjectedGradientL2Norm(gdelta);
				CompositeStopingCriteria compositeStop = new CompositeStopingCriteria();
				compositeStop.add(stopGrad);
				optimizer.setMaxIterations(maxStep);
				
				if (optimizer.optimize(testcase, new OptimizerStats(), compositeStop)) {
					ratio_a = testcase.getPA();
					ratio_b = 1.0 - ratio_a;
				}
			}

			count[0][0] += ratio_a * sample[i][0];
			count[0][1] += ratio_a * sample[i][1];
			
			count[1][0] += ratio_b * sample[i][0];
			count[1][1] += ratio_b * sample[i][1];	
		}
				
	
	}
	
	public static void main(String[] args) {
		coinTossPrOpt test = new coinTossPrOpt();
		test.em();
	}
}
