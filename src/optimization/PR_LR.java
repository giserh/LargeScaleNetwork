package optimization;

//import optimization.gradientBasedMethods.ProjectedObjective;
//import optimization.projections.BoundsProjection;
//import optimization.projections.Projection;

public class PR_LR extends ProjectedObjective {

	double m_p[];
	double[] m_phi_Z_x;
	double m_epsilon = 0.1; // slack variable
	Projection m_projection;
	
	public PR_LR(double p[], int true_label) {//HdT means the difference between head and tail 
		m_p = p;
		initiate_constraint_feature(true_label);
		parameters = new double[]{1.0};//start from a legal point
		gradient = new double[]{0.0};
		m_projection = new BoundsProjection(0.2,Double.MAX_VALUE);
	}

	public void initiate_constraint_feature(int label)
	{
		m_phi_Z_x = new double[5];
		if(label == 0)
		{
			m_phi_Z_x [0] = -5;
			if(m_p[1]<=m_p[0])
				m_phi_Z_x [1] = -4;
			else
				m_phi_Z_x [1] = +4;
			
			if(m_p[2]<=m_p[1])
				m_phi_Z_x [2] = -3;
			else
				m_phi_Z_x [2] = +3;
			
			if(m_p[3]<=m_p[2])
				m_phi_Z_x [3] = -2;
			else
				m_phi_Z_x [3] = +2;
			
			if(m_p[4]<=m_p[3])
				m_phi_Z_x [4] = -1;
			else
				m_phi_Z_x [4] = 1;
		}
		
		if(label == 1)
		{
			if(m_p[0]<=m_p[1])
				m_phi_Z_x [0] = -3;
			else
				m_phi_Z_x [0] = +3;
			
			m_phi_Z_x [1] = -4;
			
			if(m_p[2]<=m_p[1])
				m_phi_Z_x [2] = -3;
			else
				m_phi_Z_x [2] = +3;
			
			if(m_p[3]<=m_p[2])
				m_phi_Z_x [3] = -2;
			else
				m_phi_Z_x [3] = +2;
			
			if(m_p[4]<=m_p[3])
				m_phi_Z_x [4] = -1;
			else
				m_phi_Z_x [4] = +1;
		}
		
		if(label == 2)
		{
			
			if(m_p[0]<=m_p[1])
				m_phi_Z_x [0] = -1;
			else
				m_phi_Z_x [0] = +1;
			
			if(m_p[1]<=m_p[2])
				m_phi_Z_x [1] = -2;
			else
				m_phi_Z_x [1] = 2;
			
			m_phi_Z_x [2] = -3;
			
			if(m_p[3]<=m_p[2])
				m_phi_Z_x [3] = -2;
			else
				m_phi_Z_x [3] = 2;
			
			if(m_p[4]<=m_p[3])
				m_phi_Z_x [4] = -1;
			else
				m_phi_Z_x [4] = +1;
		}
		if(label == 3)
		{
			if(m_p[0]<=m_p[1])
				m_phi_Z_x [0] = -1;
			else
				m_phi_Z_x [0] = +1;
			
			if(m_p[1]<=m_p[2])
				m_phi_Z_x [1] = -2;
			else
				m_phi_Z_x [1] = +2;
			
			if(m_p[2]<=m_p[3])
				m_phi_Z_x [2] = -3;
			else
				m_phi_Z_x [2] = +3;
			
			m_phi_Z_x [3] = -4;// should it be -2
			
			if(m_p[4]<=m_p[3])
				m_phi_Z_x [4] = -3;
			else
				m_phi_Z_x [4] = +3;
		}
		if(label == 4)
		{
			if(m_p[0]<=m_p[1])
				m_phi_Z_x [0] = -1;
			else
				m_phi_Z_x [0] = +1;
			
			if(m_p[1]<=m_p[2])
				m_phi_Z_x [1] = -2;
			else
				m_phi_Z_x [1] = +2;
			
			if(m_p[2]<=m_p[3])
				m_phi_Z_x [2] = -3;
			else
				m_phi_Z_x [2] = +3;
			
			if(m_p[3]<=m_p[4])
				m_phi_Z_x [3] = -4;
			else
				m_phi_Z_x [3] = +4;
			
			
			m_phi_Z_x [4] = -5;
		}
	}
	
	@Override
	public double[] getGradient() {
		gradientCalls ++;
		double z_lambda = 
				  m_p[0]*Math.exp(-parameters[0]*m_phi_Z_x[0]) 
				+ m_p[1]*Math.exp(-parameters[0]*m_phi_Z_x[1])
				+ m_p[2]*Math.exp(-parameters[0]*m_phi_Z_x[2])
				+ m_p[3]*Math.exp(-parameters[0]*m_phi_Z_x[3])
				+ m_p[4]*Math.exp(-parameters[0]*m_phi_Z_x[4]);
		
		gradient[0] = 
				- m_p[0]*Math.exp(-parameters[0]*m_phi_Z_x[0])*m_phi_Z_x[0]/z_lambda 
				- m_p[1]*Math.exp(-parameters[0]*m_phi_Z_x[1])*m_phi_Z_x[1]/z_lambda
				- m_p[2]*Math.exp(-parameters[0]*m_phi_Z_x[2])*m_phi_Z_x[2]/z_lambda
				- m_p[3]*Math.exp(-parameters[0]*m_phi_Z_x[3])*m_phi_Z_x[3]/z_lambda
				- m_p[4]*Math.exp(-parameters[0]*m_phi_Z_x[4])*m_phi_Z_x[4]/z_lambda
				+ 2*m_epsilon*parameters[0];
		return gradient;
	}

	@Override
	public double getValue() {
		functionCalls++;
		double z_lambda = 
				  m_p[0]*Math.exp(-parameters[0]*m_phi_Z_x[0]) 
				+ m_p[1]*Math.exp(-parameters[0]*m_phi_Z_x[1])
				+ m_p[2]*Math.exp(-parameters[0]*m_phi_Z_x[2])
				+ m_p[3]*Math.exp(-parameters[0]*m_phi_Z_x[3])
				+ m_p[4]*Math.exp(-parameters[0]*m_phi_Z_x[4]);
		return Math.log(z_lambda) + m_epsilon*parameters[0]*parameters[0];
	}
	
	@Override
	public double[] projectPoint(double[] point) {
		double[] newPoint = point.clone();
		m_projection.project(newPoint);
		return newPoint;
	}
	
	public double[] getPA() {
		double z_lambda = 
				  m_p[0]*Math.exp(-parameters[0]*m_phi_Z_x[0]) 
				+ m_p[1]*Math.exp(-parameters[0]*m_phi_Z_x[1])
				+ m_p[2]*Math.exp(-parameters[0]*m_phi_Z_x[2])
				+ m_p[3]*Math.exp(-parameters[0]*m_phi_Z_x[3])
				+ m_p[4]*Math.exp(-parameters[0]*m_phi_Z_x[4]);
		
		double tmp_m_p [] =new double[5];
		tmp_m_p[0] = m_p[0]*Math.exp(-parameters[0]*m_phi_Z_x[0]) / z_lambda;
		tmp_m_p[1] = m_p[1]*Math.exp(-parameters[0]*m_phi_Z_x[1]) / z_lambda;
		tmp_m_p[2] = m_p[2]*Math.exp(-parameters[0]*m_phi_Z_x[2]) / z_lambda;
		tmp_m_p[3] = m_p[3]*Math.exp(-parameters[0]*m_phi_Z_x[3]) / z_lambda;
		tmp_m_p[4] = m_p[4]*Math.exp(-parameters[0]*m_phi_Z_x[4]) / z_lambda;
		return tmp_m_p;
	}

	@Override
	public String toString() {
		return "LogisticRegression";
	}
}
