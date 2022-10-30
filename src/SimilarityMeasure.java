package nini.nlp;

public class SimilarityMeasure {
	
    private String measure;
    
    public SimilarityMeasure(String measure) {
        this.setMeasure(measure);
    }

    public SimilarityMeasure() {
        this(null);
    }
    
    public double score(double[] a, double[] b) {
        return score(a, b, this.measure);
    }
    
    public double score(double[] a, double[] b, String measure) {
    	
        switch (measure) {
            case "cosine":
            	return cosineSimilarity(a, b);
            	
            case "euclidean":
            	return euclideanDist(a, b);
            	
            default:
            	return cosineSimilarity(a, b);
        }
    }

    private void setMeasure(String measure) {
        String defaultMeasure = "cosine";

        if (measure == null) {
            this.measure = defaultMeasure;
            return;
        }
        else {
        	this.measure = measure;
        }
    }

    public double euclideanDist(double[] a, double[] b) {
        
        double sum = 0;
        
        for (int i = 0; i < a.length; i++)
        	sum += Math.pow(b[i] - a[i],2);
  
        return Math.sqrt(sum);
    }
    
    public double cosineSimilarity(double[] a, double[] b) {
    	
        double dotProduct = 0.0;
        double normalizedA = 0.0, normalizedB = 0.0;
        
        for (int i = 0; i < a.length; i++) {
        	dotProduct += a[i] * b[i];
            normalizedA += Math.pow(a[i], 2);
            normalizedB += Math.pow(b[i], 2);
        }
        return dotProduct / (Math.sqrt(normalizedA) * Math.sqrt(normalizedB));
    }
}
