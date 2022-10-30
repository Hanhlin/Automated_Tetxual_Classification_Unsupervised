package nini.nlp;

import java.util.Arrays;

public class KmeansPP extends Kmeans{

	public KmeansPP(double[][] matrix, int K) {
		super(matrix, K);
	}
	
	@Override
	public void cluster(int iterNum) {
    	
        for (int i = 0; i < iterNum; i++) {
        	
            System.out.println("Iteration " + i);
            System.out.println(Arrays.toString(this.clusterIter(i)));
            
            double purity = getPurity();
            System.out.println("Purity: " + purity + "\n");
            
            if (purity == 1.0)
                break;
        }
    }
    
	
	public int[] clusterIter(int iter) {

		if (iter == 0)
			this.centroids = this.getSmarterCentroids();
		else
			this.centroids = this.getRandomCentroids();
        this.clusterAssignments = new int[this.docNum];
        this.prevAssignments = new int[this.docNum];

        // cluster assignment
        for (int epoch = 0; epoch < this.epochsNum; epoch++) {
            
            for (int i = 0; i < this.docNum; i++) {
                this.clusterAssignments[i] = this.getClosestCluster(this.matrix[i]);
            }
            this.updateClusters();

            if (DEBUG) {
                if (epoch > this.epochsNum - 2)
                    System.out.println(Arrays.toString(clusterAssignments));
            }
            if (this.stopKMeans())
                break;
        }
        return this.clusterAssignments;
    }


}
