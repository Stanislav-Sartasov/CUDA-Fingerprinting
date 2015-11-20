using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NonParrallelVersion
{
    class ModifiedJangSuen
    {       
        private int countA1(int[,] A)
        {
            int result = 0;

            for (int k = 1; k <= 2; k++)
            {
                if ((A[0, k] == 1) && (A[0, k - 1] == 0))
                    result++;
            }

            for (int l = 1; l <= 2; l++)
            {
                if ((A[l, 2] == 1) && (A[l - 1, 2] == 0))
                    result++;
            }

            for (int k = 1; k >= 0; k--)
            {
                if ((A[2, k] == 1) && (A[2, k + 1] == 0))
                    result++;
            }

            for (int l = 1; l >= 0; l--)
            {
                if ((A[l, 0] == 1) && (A[l + 1, 0] == 0))
                    result++;
            }
            return result;
        }

        private int countB1(int[,] A)
        {
            int result = 0;
            for (int k = 0; k < 3; k++)
            {
                for (int l = 0; l < 3; l++)
                {
                    if ((A[k, l] == 1) && ((k != 1) || (l != 1)))
                        result++;
                }
            }
            return result;
        }

        public int[,] JangSuen(int[,] A, int n, int m)
        {
            int B1 = 0;
            int A1 = 0;

            for (int i = 1; i < n - 1; i++)
            {
                for (int j = 1; j < m - 1; j++)
                {
                    if (A[i, j] == 1)
                    {
                        int[,] P = new int[3, 3] 
                                                {{ A[i - 1, j - 1], A[i - 1, j], A[i - 1, j + 1]},
                                                 { A[i, j - 1], A[i, j], A[i, j + 1]},
                                                 { A[i + 1, j - 1], A[i + 1, j], A[i + 1, j + 1]}};

                        bool a1 = ((P[0, 0] * P[0, 2] * P[2, 2] == 0) || (P[0, 2] * P[2, 2] * P[2, 0] == 0)) && (i % 2 == 1); //P2*P4*P6 = 0 или P4*P6*P8 = 0
                        bool b1 = ((P[0, 0] * P[0, 2] * P[2, 0] == 0) || (P[0, 0] * P[2, 2] * P[2, 0] == 0)) && (i % 2 == 0);
                        bool a2 = ((P[0, 2] * P[2, 2] == 1 && P[1, 0] == 0) || (P[0, 2] * P[0, 0] == 1 && (1 - P[0, 1]) * (1 - P[2, 1]) * (1 - P[2, 0]) == 1)) && (i % 2 == 1);
                        bool b2 = ((P[0, 0] * P[2, 0] == 0 && P[1, 2] == 0) || (P[2, 2] * P[2, 0] == 1 && (1 - P[0, 1]) * (1 - P[2, 1]) * (1 - P[0, 2]) == 1)) && (i % 2 == 0);

                        B1 = countB1(P);
                        A1 = countA1(P);

                        if ((B1 <= 6) && (B1 >= 2))
                        {
                            if ((A1 == 1) && (a1 || b1))
                            {
                                A[i, j] = 0;
                            }
                        }

                        else
                            if ((A1 == 2) && (a2 || b2))
                            {
                                A[i, j] = 0;
                            }
                    }
                }
            }
            return A;
        }
    }
}
