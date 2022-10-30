package nini.nlp;

import java.util.*;

public class Main {

	public static void main(String[] args) throws Exception {
		
		// ====== pre-process documents, generate term-freq matrix, tf-idf matrix ======
		
        List<String> documentStrings = preProcess.readData("train");
        List<List<String>> termsList = preProcess.vectorize(documentStrings, "stopwords.txt");

        Map<String, Integer> termIndex = preProcess.getTermsIndex(termsList);
        double[][] tf = preProcess.getTermFreqMatrix(termsList, termIndex);
        //printInfo(tf, "Term-frequency Matrix:");
        
        double[][] tf_idf = preProcess.getTfIdf(tf);
        //printInfo(tf_idf, "TF-IDF Matrix:");
 
        
        // ========== generate keywords per folder ==========
        
        preProcess.getTopTerms(tf_idf, termIndex, "topics.txt");
//        Kmeans model = new Kmeans(tf_idf, 3);
        KmeansPP model = new KmeansPP(tf_idf, 3);
        model.cluster();

        // ========== read test data ========== 
        
        List<String> tests =  preProcess.readData("test");
        double[][] test_tf_idf = preProcess.getTest_TfIdf(tf, termIndex, tests, "stopwords.txt");
        int[] clusterAssignment = preProcess.predictCluster(model.centroids, test_tf_idf);
        
        // ========== maintain predicted cluster label each doc ==========
        
        ArrayList<List<Integer>> clustersList = new ArrayList<>(3);
        for (int k = 0; k < 3; k++) {
            List<Integer> temp_cluster = new ArrayList<>();
            for (int doc_ind = 0; doc_ind < test_tf_idf.length; doc_ind++) {
                if (clusterAssignment[doc_ind] == k) {
                    temp_cluster.add(doc_ind);
                }
            }
            clustersList.add(temp_cluster);
        }

        // ========== Predicted Cluster Evaluation ==========
        
        ClusterEvaluation ce = new ClusterEvaluation(clustersList, "confusion matrix.txt");
        printInfo(ce);
        
        // ========== reduce dimensionality by SVD ==========
        
        SVD svd = new SVD();
        double[][] svd_components = svd.svd(tf_idf, 2);
        printInfo(svd_components, "SVD Matrix:");
        
        
        // ========== Cluster Visualization ==========
        
        Visualizer v = new Visualizer(svd_components, model.clusterAssignments, "Original Cluster Plot");
        v.setData();
        v.frame.setVisible(true);

        Visualizer v2 = new Visualizer(svd.svd(test_tf_idf, 2), clusterAssignment, "Test Cluster Plot");
        v2.setData();
        v2.frame.setVisible(true);
		
	}
	
	public static void printInfo(ClusterEvaluation ce) {
		
		System.out.println("Precisions:");
		System.out.println(ce.precisions + "\n");

		System.out.println("Recalls:");
		System.out.println(ce.recalls + "\n");

		System.out.println("F1 Scores:");
		System.out.println(ce.F1_scores + "\n");
  
	}
	
	public static void printInfo(double[][] matrix, String title) {
		
		System.out.println(title);
        for (double[] row : matrix) {
            System.out.println(Arrays.toString(row));
        }
        System.out.println("\n");
	}

}
