using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
//using System.Numerics;

namespace CUDAFingerprinting.Common
{
    public class NovelSinglePass
    {
        private int width;
        private int height;
        public NovelSinglePass(double[,] dAr)
        {
            width = dAr.GetLength(0);
            height = dAr.GetLength(1);
        }

        
        public double PN(double[,] P)
        {
            double res = 0;

            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    if (!(i == 1 && j == 1))
                        res += P[i, j];

            return res;
        }

        public double CN(double[,] P, double [,] Q)
        {
            double res = 0;

            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    if (!(i == 1 && j == 1))
                        res += P[i, j] * Q[i, j];

            return res;
        }

        public double count(double[,] P, double[,] Q, int i, int j)
        {
            double val1, val2;
            if (j != 2)
            {
                val1 = P[i, j];
                val2 = Q[i, j];
            }
            else
            {
                val1 = P[i + 1, 0];
                val2 = Q[i + 1, 0];
            }
            if (i == 2 && j == 2)
            {
                val1 = P[0, 0];
                val2 = Q[0, 0];
            }

            if (P[i, j] * Q[i, j] == 0 && val1 * val2 == 1)
                return 1;
            return 0;
        }

        public double Trans(double[,] P, double[,] Q)
        {
            double res = 0;

            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    if (!(i == 1 && j == 1))
                        res += count(P, Q, i, j);

            return res;
        }
        public double[,] Thinning(double[,] img)
        {
            double[,] Q = new double[height, width];
            double[,] miniP = new double[3, 3];
            double[,] miniQ = new double[3, 3];

            for (int i = 0; i < height; ++i)
                for (int j = 0; j < width; ++j)
                    Q[i, j] = 1;

            for (int i = 0; i < height; i += 3)
                for (int j = 0; j < width; j += 3)
                {
                    for ( int k = 0; k < 3; ++k )
                        for ( int l = 0; l < 3; ++l )
                        {
                            miniP[k, l] = img[i + k, j + l];
                            miniQ[k, l] = Q[i + k, j + l];
                        }
                }
            
            ...
            //return ;
        }

        public Bitmap MakeBitmap(double[,] byteMatrix)
        {
            Bitmap bmp = new Bitmap(height, width);

            for (int x = 0; x < height; ++x)
            {
                for (int y = 0; y < width; ++y)
                {
                    var value = (int)(255*byteMatrix[x, y]);
                    bmp.SetPixel(x, y, Color.FromArgb(255, value, value, value));
                }
            }

            return bmp;
        }

        public void SaveSegmentation(Bitmap bmp, string filename)
        {
            bmp.Save(filename);
        }
    }
}
