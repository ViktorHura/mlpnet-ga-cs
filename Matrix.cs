using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Medallion;

namespace mlp_net_ga_cs
{
    class Matrix
    {
        public double[,] matrix;

        public void RandomMatrix(ref Random rnd,in int rows, in int cols)
        {
            
            this.matrix = new double[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    this.matrix[i, j] = rnd.NextGaussian();
                }
            }
        }

        public void EmptyMatrix(in int rows,in int cols)
        {
            
            this.matrix = new double[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    this.matrix[i, j] = 0.0;
                }
            }
        }

        public Matrix DotProduct(in Matrix mat)
        {
            int aRows = this.matrix.GetLength(0);
            int aCols = this.matrix.GetLength(1);
            int bRows = mat.matrix.GetLength(0);
            int bCols = mat.matrix.GetLength(1);

            if (aCols != bRows)
            {
                throw new System.ArgumentException("A:Cols: " + aCols + " did not match B:Rows " + bRows + ".", "");
            }

            double[,] C = new double[aRows, bCols];
            for (int i = 0; i < aRows; i++)
            {
                for (int j = 0; j < bCols; j++)
                {
                    C[i, j] = 0.0;
                }
            }

            for (int i = 0; i < aRows; i++)
            {
                for (int j = 0; j < bCols; j++)
                {
                    for (int k = 0; k < aCols; k++)
                    {
                        C[i, j] += this.matrix[i, k] * mat.matrix[k, j];
                    }
                }
            }

            Matrix product = new Matrix();
            product.matrix = C;
            return product;
        }

        public Matrix Sigmoid()
        {
            for (int i = 0; i < this.matrix.GetLength(0); i++)
            {
                for (int j = 0; j < this.matrix.GetLength(1); j++)
                {
                    this.matrix[i, j] = sigmoid(this.matrix[i, j]);
                }
            }

            return this;
        }

        private static double sigmoid(double x)
        {
            return (1.0 / (1.0 + Math.Exp(-x)));
        }

        public static void printMatrix(Matrix mat)
        {
            int rowLength = mat.matrix.GetLength(0);
            int colLength = mat.matrix.GetLength(1);

            for (int i = 0; i < rowLength; i++)
            {
                for (int j = 0; j < colLength; j++)
                {
                    Console.Write(string.Format("{0} ", mat.matrix[i, j]));
                }
                Console.Write(Environment.NewLine);
            }
            
        }

        public int Rows()
        {

            return this.matrix.GetLength(0);
        }
        public int Cols()
        {

            return this.matrix.GetLength(1);
        }
    }
}
