package nini.nlp;

import Jama.Matrix;
import Jama.SingularValueDecomposition;

public class SVD {
    public double [][] svd(double [][] matrix, int n) {
        Matrix m = new Matrix(matrix);
        SingularValueDecomposition svd = m.svd();
        
        Matrix U = svd.getU();
        double[][] U_Matrix = new double[matrix.length][n];
        for (int i = 0; i < U_Matrix.length; i++) {
            for (int j = 0; j < n; j++) {
                U_Matrix[i][j] = U.get(i, j);
            }
        }
        return U_Matrix;
    }
}