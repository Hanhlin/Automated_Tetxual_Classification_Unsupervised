package nini.nlp;

import java.io.*;
import java.util.*;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.CoreEntityMention;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;

public class preProcess {

	public static List<String> readData(String path) throws Exception {
		
		String text;
		List<String> docs = new ArrayList<>();
		
		File fileList[] = new File(path).listFiles();
		Arrays.sort(fileList);

		for (File dir : fileList) {
			if (dir.isDirectory()) {
				File files[] = new File(dir.getPath()).listFiles();
				Arrays.sort(files);
				for(File f : files) {
					if (f.isHidden())
						continue;
					Scanner scanner = new Scanner(f);
					text = scanner.useDelimiter("\\Z").next();
					docs.add(text);
					scanner.close();
				}
			}
			else {
				if (dir.isHidden())
					continue;
				Scanner scanner = new Scanner(dir);
				text = scanner.useDelimiter("\\Z").next();
				docs.add(text);
				scanner.close();
			}
		}
		return docs;
	}
	
	public static List<List<String>> vectorize(List<String> docsStrings, String stopPath) throws Exception {

		List<List<String>> wordsList = new ArrayList<List<String>>();
        Set<String> stopWords = getStopWords(stopPath);					
		StanfordCoreNLP pipeline = initPipeline();
        List<String> tempWordsList;
        List<CoreLabel> tokenList;
		
        for (String docS : docsStrings) {

        	docS = docS.replaceAll("\\s+", " ");
        	
        	CoreDocument document = new CoreDocument(docS);
            pipeline.annotate(document);

            tempWordsList = new ArrayList<String>();
            tokenList = document.tokens();
            
            for (CoreLabel token : tokenList) {
                String t = token.lemma().toLowerCase();
                
                if (!stopWords.contains(t)) {
                    if (t.replaceAll("[^a-zA-Z0-9]+", "").length() > 1)
                    	tempWordsList.add(t);
                }
            }

            //merge name entities
            Set<String> entities = new HashSet<String>();
            for (CoreEntityMention en : document.entityMentions()) {
                entities.add(en.text().toLowerCase());
            }
            wordsList.add(mergeEntities(tempWordsList, entities));
        }
        
        return wordsList;
    }
    
	public static StanfordCoreNLP initPipeline() {
    	
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit, pos, lemma, ner");
        return new StanfordCoreNLP(props);
    }
    
    public static Set<String> getStopWords(String stopPath) throws Exception {
    	
        Scanner scanner = new Scanner(new File(stopPath));
        Set<String> stopWords = new HashSet<String>();
        
        while (scanner.hasNext())
        	stopWords.add(scanner.next());
 
        scanner.close();
        
        return stopWords;
    }
	
    public static List<String> mergeEntities(List<String> wordsList, Set<String> entities) {
    	
        List<String> words = new ArrayList<>();

        for (int i = 0; i < wordsList.size(); i++) {
            boolean found = false;
            
            // try 2-grams and 3-grams
            for (int n = 2; n < 4; n++) {
                List<String> strings = new LinkedList<>();
                
                for (int j = i; j < i + n && j < wordsList.size(); j++)
                	strings.add(wordsList.get(j));
            	String name = String.join(" ", strings);
            	
                if (entities.contains(name)) {
                	words.add(name);
                    i += n;
                    found = true;
                    break;
                }
            }

            // if word is not a name entity, still add to final words list
            if (!found)
            	words.add(wordsList.get(i));
        }
        return words;
    }
        
    public static Map<String,Integer> getTermsIndex(List<List<String>> termsList) {
    	
        Map<String,Integer> termIndex = new HashMap<String,Integer>();
        
        int term_index = 0;
        for (List<String> term: termsList) {
            for (String word: term) {
                if (!termIndex.containsKey(word)) {
                	termIndex.put(word,term_index++);
                }
            }
        }
        return termIndex;
    }
    
    public static double[][] getTermFreqMatrix(List<List<String>> termsList, Map<String,Integer> termIndex) {
    	
        double[][] tf = new double[termsList.size()][termIndex.size()];

        for (int doc_index = 0; doc_index < termsList.size(); doc_index++) {
        	
        	//compute total number of each term in each doc
            for (String t: termsList.get(doc_index)) {
            	tf[doc_index][termIndex.get(t)]++;
            }

            //get term frequency matrix
            for (int term_index = 0; term_index < termIndex.size(); term_index++) {
            	tf[doc_index][term_index] = (tf[doc_index][term_index] / termsList.get(doc_index).size());
            }
        }
        return tf;
    }
    
    public static double[][] getTfIdf(double[][] tf) {
        
    	double[][] tf_idf =  new double[tf.length][tf[0].length];
        
        for (int term_index = 0; term_index < tf[0].length; term_index++) {
            double docCount = 0;
            
            for (int doc_index = 0; doc_index < tf.length; doc_index++)
                docCount += (tf[doc_index][term_index]) > 0 ? 1: 0;

            // compute tf_idf of term at term_index for each doc
            for (int doc_index = 0; doc_index < tf.length; doc_index++)
                tf_idf[doc_index][term_index] = tf[doc_index][term_index] * Math.log(tf.length / docCount);
        }
        
        // ***************** print matrix *****************
        
//        System.out.println("TF-IDF Matrix:");
//        for (double[] row : tf_idf) {
//            System.out.println(Arrays.toString(row));
//        }
//        System.out.println("\n");
        
        // ************************************************
        
        return tf_idf;
    }
    
    public static double[][] getTfIdf(double[][] tf, double[][] test_tf) {
    	
        double[][] test_tf_idf =  new double[test_tf.length][test_tf[0].length];
        
        for (int term_index = 0; term_index < tf[0].length; term_index++) {
            double docCount = 0;
            
            for (int doc_index = 0; doc_index < tf.length; doc_index++)
            	docCount += tf[doc_index][term_index] > 0 ? 1: 0;

            for (int doc_index = 0; doc_index < test_tf.length; doc_index++)
                test_tf_idf[doc_index][term_index] = test_tf[doc_index][term_index] * Math.log(tf.length / docCount);
        }
        return test_tf_idf;
    }

	public static void getTopTerms(double[][] tf_idf, Map<String,Integer> termIndex, String writePath) throws IOException {

		File myObj = new File(writePath);
        FileWriter myWriter = new FileWriter(myObj);
        
        System.out.println("************** Topics per Folder **************\n");
        
        int doc_index = 0;
        for (int k = 1; k <= 7; k += 3) {
            double[] counts = new double[termIndex.size()];
            for (int i = doc_index; i < doc_index + 8 && i < tf_idf.length; i++) {
                for (int term_index = 0; term_index < termIndex.size(); term_index++) {
                    counts[term_index] += tf_idf[i][term_index];
                }
            }
            doc_index += 8;
            
            List<String> opa = new ArrayList<String>(termIndex.keySet());
            opa.sort(Comparator.comparingDouble(o -> counts[termIndex.get(o)]));
            System.out.println("Folder: C" + k );
            System.out.println(opa.subList(opa.size() - 5, opa.size()) + "\n");

            //generate topics per Folder in write path
            myWriter.write("Folder: C" + k + "\n");
            myWriter.write(opa.subList(opa.size() - 5, opa.size()).toString() + "\n\n");
        }
        
        System.out.println("***********************************************\n");
        
        myWriter.close();
    }
	    
	public static double[][] getTest_TfIdf(double[][] tf, Map<String,Integer> termIndex, List<String> tests, String stopPath) throws Exception{
       
		List<List<String>> wordsList = vectorize(tests, stopPath);
        int testNum = wordsList.size();
        
        double[][] test_tf = new double[testNum][termIndex.size()];
        
        for (int doc_index = 0; doc_index < testNum; doc_index++) {
            for (String word: wordsList.get(doc_index)) {
                if (termIndex.containsKey(word)) {
                	test_tf[doc_index][termIndex.get(word)] += 1;
                }
            }
        }

        return getTfIdf(tf, test_tf);
    }
     
    public static int[] predictCluster(double[][] centroids,double[][] test_tf_idf) {
        
    	SimilarityMeasure sm = new SimilarityMeasure();
        int[] clusterAssign = new int[10];

        for (int i = 0; i < test_tf_idf.length; i++) {
        	
            Map<Integer,Double> scores = new HashMap<>(centroids.length);
        
            for (int j = 0; j < centroids.length; j++)
                scores.put(j, sm.score(test_tf_idf[i],centroids[j]));
            
            List<Map.Entry<Integer, Double>> centroidScores = new LinkedList<>(scores.entrySet());
            
            Collections.sort(centroidScores, new Comparator<>() {
                public int compare(Map.Entry<Integer, Double> c1, Map.Entry<Integer, Double> c2) {
                    return (c1.getValue()).compareTo(c2.getValue());
                }
            });
            
            clusterAssign[i] = centroidScores.get(0).getKey();
        }
        
        // ***************** print prediction *************
        
        List<String> predictedCluster = new ArrayList<>();
        for (int i : clusterAssign) {
        	predictedCluster.add(i + "");
        }
        System.out.println("Cluster Prediction:");
        System.out.println(predictedCluster + "\n");
        
        // ************************************************
        
        return clusterAssign;
    }
    
}
