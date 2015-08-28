using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TriangulationWithAfineTransformation.Classes
{
    class LinearSystem
    {
        public static double[] SolveLinearSystem(double[,] matrix, double[] result)
        {
            double[] x = new double[result.Length];

            for (int i = 0; i < result.Length; i++)
            {
                for (int j = i; j < result.Length; j++)
                    if (Math.Abs(matrix[j, i]) > Math.Abs(matrix[i, i]))
                    {
                        double tmp;
                        for (int k = i; k < result.Length; k++)
                        {
                            tmp = matrix[i, k];
                            matrix[i, k] = matrix[j, k];
                            matrix[j, k] = tmp;
                        }
                        tmp = result[i];
                        result[i] = result[j];
                        result[j] = tmp;
                    }
                for (int j = i + 1; j < result.Length; j++)
                {
                    double koef = matrix[j, i] / matrix[i, i];
                    for (int k = i; k < result.Length; k++)
                        matrix[j, k] -= matrix[i, k] * koef;
                    result[j] -= result[i] * koef;
                }
            }
            for (int i = result.Length - 1; i > -1; i--)
            {
                double sum = 0;
                for (int k = result.Length - 1; k > i; k--)
                    sum += matrix[i, k] * x[k];
                x[i] = (result[i] - sum) / matrix[i, i];
            }
            return x;
        }
    }
}
