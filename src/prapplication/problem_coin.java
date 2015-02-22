package prapplication;

import optimization.gradientBasedMethods.ProjectedObjective;
import optimization.projections.BoundsProjection;
import optimization.projections.Projection;

public class problem_coin extends ProjectedObjective {

	double m_pA;
	double[] m_phi_Z_x;
	double m_epsilon = 0.1; // slack variable
	Projection m_projection;
	
	public problem_coin(double pA, int HdT) {//HdT means the difference between head and tail 
		m_pA = pA;
		m_phi_Z_x = new double[2];//more heads: 1; more tails: -1; equal 0
		if (HdT>0) {
			m_phi_Z_x[0] = -1;
			m_phi_Z_x[1] = 1;
		} else if (HdT<0) {
			m_phi_Z_x[0] = 1;
			m_phi_Z_x[1] = -1;
		}
		
		parameters = new double[]{1.0};//start from a legal point
		gradient = new double[]{0.0};
		m_projection = new BoundsProjection(0.2,Double.MAX_VALUE);
	}

	@Override
	public double[] getGradient() {
		gradientCalls ++;
		double z_lambda = m_pA*Math.exp(-parameters[0]*m_phi_Z_x[0]) + (1.0-m_pA)*Math.exp(-parameters[0]*m_phi_Z_x[1]);
		gradient[0] = - m_pA*Math.exp(-parameters[0]*m_phi_Z_x[0])*m_phi_Z_x[0]/z_lambda 
				- (1.0-m_pA)*Math.exp(-parameters[0]*m_phi_Z_x[1])*m_phi_Z_x[1]/z_lambda
				+ 2*m_epsilon*parameters[0];;
		return gradient;
	}

	@Override
	public double getValue() {
		functionCalls++;
		double z_lambda = m_pA*Math.exp(-parameters[0]*m_phi_Z_x[0]) + (1.0-m_pA)*Math.exp(-parameters[0]*m_phi_Z_x[1]);
		return Math.log(z_lambda) + m_epsilon*parameters[0]*parameters[0];
	}
	
	@Override
	public double[] projectPoint(double[] point) {
		double[] newPoint = point.clone();
		m_projection.project(newPoint);
		return newPoint;
	}
	
	public double getPA() {
		double z_lambda = m_pA*Math.exp(-parameters[0]*m_phi_Z_x[0]) + (1.0-m_pA)*Math.exp(-parameters[0]*m_phi_Z_x[1]);
		return m_pA*Math.exp(-parameters[0]*m_phi_Z_x[0]) / z_lambda;
	}

	@Override
	public String toString() {
		return "ProblemCoin";
	}
}
