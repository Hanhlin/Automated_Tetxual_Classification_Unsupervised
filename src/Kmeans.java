package nini.nlp;

import java.util.*;

public class Kmeans {
	
    protected final boolean DEBUG = false;

    public double[][] matrix, centroids;
    public int K, docNum, wordNum, epochsNum;
    public int[] clusterAssignments;
    protected int[] prevAssignments;
    private SimilarityMeasure similarityMeasure;
    
    public Kmeans(double[][] matrix, int K) {
    	
        this.K = K;
        this.matrix = matrix;
        this.epochsNum = 500;
        this.docNum = this.matrix.length;
        this.wordNum = this.matrix[0].length;
        this.similarityMeasure = new SimilarityMeasure();

    }

    public void cluster(int iterNum) {
    	
        for (int i = 0; i < iterNum; i++) {
        	
            System.out.println("Iteration " + i);
            System.out.println(Arrays.toString(this.clusterIter()));
            
            double purity = getPurity();
            System.out.println("Purity: " + purity + "\n");
            
            if (purity == 1.0)
                break;
        }
    }
    
    public void cluster() {
        this.cluster(100);
    }
    
    public int[] clusterIter() {

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

    public int predictLabel(double[] x) {
        return this.getClosestCluster(x);
    }

    public double getPurity() {
    	
        double purity = 0;
        HashSet<Integer> seenCluster = new HashSet<>();
        
        for (int k = 0; k < this.K; k++) {
            HashMap<Integer, Integer> counts = new HashMap<Integer, Integer>();
            
            for (int i = 8 * k; i < 8 * (k + 1); i++) {
                counts.put(this.clusterAssignments[i], counts.getOrDefault(this.clusterAssignments[i], 0) + 1);
            }
            
            int maxC = -1, maxV = -1;
            for (Integer c : counts.keySet()) {
                if (counts.get(c) > maxV) {
                    maxC = c;
                    maxV = counts.get(c);
                }
            }
            
            if (seenCluster.contains(maxC)) 
            	continue;
            seenCluster.add(maxC);
            purity += maxV;
        }

        return purity / 24.0;

    }

    public boolean stopKMeans() {
    	
        int diffNum = 0;
        
        for (int i = 0; i < this.docNum; i++) {
            if (this.clusterAssignments[i] != this.prevAssignments[i]) {
            	diffNum += 1;
            }
        }
        if (diffNum == 0) return true;

        for (int i = 0; i < this.docNum; i++) {
            this.prevAssignments[i] = this.clusterAssignments[i];
        }
        return false;

    }

    public void updateClusters() {
    	
        for (int k = 0; k < this.K; k++) {
        	
            double clusterSize = 0;
            double[] w = new double[this.wordNum];

            for (int i = 0; i < this.docNum; i++) {
                if (k != this.clusterAssignments[i])
                    continue;
                
                clusterSize += 1;
                for (int j = 0; j < this.wordNum; j++)
                    w[j] += this.matrix[i][j];
            }
            
            if (clusterSize == 0) continue;

            for (int j = 0; j < this.wordNum; j++)
                this.centroids[k][j] = w[j] / clusterSize;
        }

    }

    public int getClosestCluster(double[] doc) {
    	
        double minDist = Double.POSITIVE_INFINITY;
        int minCluster = 0;
        
        // get the closest cluster from 3 centers for each doc
        for (int j = 0; j < this.K; j++) {
            double dist = this.similarityMeasure.score(doc, this.centroids[j]);
            
            if (dist < minDist) {
                minCluster = j;
                minDist = dist;
            }
        }
        return minCluster;
    }

    public double[][] getSmarterCentroids() {
    	
        double[][] centroids = new double[this.K][this.wordNum];
        double[] dists =new double[this.docNum];
        
        for (int k = 0; k < this.K; k++) {
            for (int i = 0; i < this.docNum; i++) {
                if (dists[i] < 0)
                    continue;
                for (int j = 0; j < this.docNum; j++) {

                    dists[i] += this.similarityMeasure.score(this.matrix[i], this.matrix[j]);
                }
            }
            int maxIndex = 0;
            double maxSum = 0;
            for (int i = 0; i < this.docNum; i++) {
                if (dists[i] > maxSum) {
                    maxIndex = i;
                    maxSum = dists[i];
                }
            }

            for (int j = 0; j < this.wordNum; j++)
                centroids[k][j] = this.matrix[maxIndex][j];

            dists[maxIndex] = -1;
        }

        return centroids;
    }

    public double[][] getRandomCentroids() {
    	
        HashSet<Integer> indices = new HashSet<Integer>();
        double[][] randomCentroids = new double[this.K][this.wordNum];
        
        while (indices.size() < this.K) {
            int rnd = new Random().nextInt(this.docNum);
            indices.add(rnd);
        }
        
        int k = 0;
        for (int doc_index: indices) {
            for (int term_index = 0; term_index < this.wordNum; term_index++)
                randomCentroids[k][term_index] = this.matrix[doc_index][term_index];
            k += 1;
        }

        return randomCentroids;
    }

}