package nini.nlp;

import java.awt.Color;
import java.util.*;

import javax.swing.JFrame;
import de.erichseifert.gral.data.DataTable;
import de.erichseifert.gral.plots.XYPlot;
import de.erichseifert.gral.ui.InteractivePanel;

public class Visualizer {

    double[][] matrix;
    int[] clusterAssignment;
    String title;
    JFrame frame;

    public Visualizer(double [][] matrix, int[] clusterAssignment, String title) {
        frame = new JFrame(title);
        this.title = title;
        this.matrix = matrix;
        this.clusterAssignment = clusterAssignment;
    }

    @SuppressWarnings({ "unchecked", "static-access" })
	public void setData() {
    	
        frame.setDefaultCloseOperation(frame.EXIT_ON_CLOSE);
        frame.setSize(700, 500);

        DataTable data1 = new DataTable(Double.class, Double.class);
        DataTable data2 = new DataTable(Double.class, Double.class);
        DataTable data3 = new DataTable(Double.class, Double.class);

        retrieve_Data(data1,0);
        retrieve_Data(data2,1);
        retrieve_Data(data3,2);

        XYPlot p = new XYPlot(data1, data2, data3);
        p.getPointRenderers(data1).get(0).setColor(Color.BLUE);
        p.getPointRenderers(data2).get(0).setColor(Color.RED);
        p.getPointRenderers(data3).get(0).setColor(Color.GREEN);

        InteractivePanel panel = new InteractivePanel(p);
        panel.setName("Cluster Plot");
        frame.setContentPane(panel);
    }

    public void retrieve_Data(DataTable data, int k) {
    	
    	List<Double> x = new ArrayList<>();
        List<Double> y = new ArrayList<>();

        for (int i = 0; i < matrix.length; i++) {
            if (clusterAssignment[i] == k) {
                x.add(matrix[i][0]);
                y.add(matrix[i][1]);
            }
        }
        for (int i = 0; i < x.size(); i++) {
            data.add(x.get(i), y.get(i));
        }
    }
}