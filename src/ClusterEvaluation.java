package nini.nlp;

import java.io.*;
import java.util.*;

public class ClusterEvaluation {
    
	ArrayList<Double> precisions, recalls, F1_scores;
	int c1_tp = 0, c1_fp = 0, c1_tn = 0, c1_fn = 0;
    int c4_tp = 0, c4_fp = 0, c4_tn = 0, c4_fn = 0;
    int c7_tp = 0, c7_fp = 0, c7_tn = 0, c7_fn = 0;
    
    public ClusterEvaluation(ArrayList<List<Integer>> clusters, String writePath) throws IOException {
    	
    	getConfusionMatrix(clusters, writePath);
    	precisions = getPrecisions();
        recalls = getRecalls();
        F1_scores = getF1scores();
    }

    public void getConfusionMatrix(ArrayList<List<Integer>> clusters, String writePath) throws IOException {
    	
    	File myObj = new File(writePath);
    	FileWriter myWriter = new FileWriter(myObj);
      
    	// ******* maintain ACTUAL cluster label *******
    	
      	List<Integer> c1 = new ArrayList<>();
		List<Integer> c4 = new ArrayList<>();
		List<Integer> c7 = new ArrayList<>();
		ArrayList<List<Integer>> temp_clusters = new ArrayList<>();
		
		int idx = 0;
		for(int i = idx; i < idx + 2; i++) {
		    c1.add(i);
		}
		
		idx += 2;
		for(int i = idx; i < idx + 6; i++) {
		    c4.add(i);
		}
		
		idx += 6;
		for(int i = idx; i < idx + 2; i++) {
		    c7.add(i);
		}
		
		temp_clusters.add(c1);
		temp_clusters.add(c4);
		temp_clusters.add(c7);
		
		// ******* maintain PREDICTED cluster label *******

		List<Integer> predictedC1 = clusters.get(0);
		List<Integer> predictedC4 = clusters.get(1);
		List<Integer> predictedC7 = clusters.get(2);
		
		// ******************* Cluster1 *******************
		
		for (int i : predictedC1) {
		    if(c1.contains(i))
		        c1_tp++;
		    else
		        c1_fp++;
		}
		c1_fn = 2 - c1_tp;
		c1_tn = 10 - (c1_tp + c1_fp + c1_fn);
		
		myWriter.write("[C1]\n");
		myWriter.write(" " + c1_tp + " " + c1_fn + "\n");
		myWriter.write(" " + c1_fp + " " + c1_tn + "\n\n");
		
		// ******************* Cluster4 *******************
		
		for (int i : predictedC4) {
		    if(c4.contains(i))
		        c4_tp++;
		    else
		        c4_fp++;
		}
		c4_fn = 6 - c4_tp;
		c4_tn = 10 - (c4_tp + c4_fp + c4_fn);
		
		myWriter.write("[C4]\n");
		myWriter.write(" " + c4_tp + " " + c4_fn + "\n");
		myWriter.write(" " + c4_fp + " " + c4_tn + "\n\n");
		
		// ******************* Cluster7 *******************
		
		for(int i : predictedC7) {
		    if(c7.contains(i))
		        c7_tp++;
		    else
		        c7_fp++;
		}
		c7_fn = 2 - c7_tp;
		c7_tn = 10 - (c7_tp + c7_fp + c7_fn);
		
		myWriter.write("[C7]\n");
		myWriter.write(" " + c7_tp + " " + c7_fn + "\n");
		myWriter.write(" " + c7_fp + " " + c7_tn + "\n\n");
		myWriter.close();
    }

	public ArrayList<Double> getPrecisions() {
		
        ArrayList<Double> precisions = new ArrayList<>();
        precisions.add((c1_tp)/(double)(c1_tp + c1_fp));
        precisions.add((c4_tp)/(double)(c4_tp + c4_fp));
        precisions.add((c7_tp)/(double)(c7_tp + c7_fp));

        return precisions;
    }

    public ArrayList<Double> getRecalls() {
    	
        ArrayList<Double> recalls = new ArrayList<>();
        recalls.add((double)(c1_tp) / (double)(c1_tp + c1_fn));
        recalls.add((double)(c4_tp) / (double)(c4_tp + c4_fn));
        recalls.add((double)(c7_tp) / (double)(c7_tp + c7_fn));

        return recalls;
    }

    public ArrayList<Double> getF1scores() {
    	
    	ArrayList<Double> scores = new ArrayList<>(Collections.nCopies(3, 0.0));
        if (precisions.get(0) != 0 || recalls.get(0) != 0)
        	scores.set(0, 2 * precisions.get(0) * recalls.get(1) / (precisions.get(0) + recalls.get(0)));
        
        if (precisions.get(1) != 0 || recalls.get(1) != 0)
        	scores.set(1, 2 * precisions.get(1) * recalls.get(1) / (precisions.get(1) + recalls.get(1)));
        
        if (precisions.get(2) != 0 || recalls.get(2) != 0)
        	scores.set(2, 2 * precisions.get(2) * recalls.get(2) / (precisions.get(2) + recalls.get(2)));

        return scores;
    }

}