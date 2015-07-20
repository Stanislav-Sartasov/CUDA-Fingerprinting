using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CUDAFingerprinting.Common;
using System.Drawing;
using System.Numerics;

namespace CUDAFingerprinting.Common.SingularityRegionDetection
{
    public class SingularityRegionDetection
    {
        private int width;
        private int height;
        public SingularityRegionDetection(double[,] dAr)
        {
            width = dAr.GetLength(0);
            height = dAr.GetLength(1);
        }

        public double Module(double[] v)
        {
            double sum = v[0] * v[0] + v[1] * v[1];
            return Math.Sqrt(sum);
        }

        public Complex[,] Regularize(Complex[,] cMap)
        {
            Complex[,] cNewMap = new Complex[width, height];

            for (int x = 1; x < width - 1; x++)
            {
                for (int y = 1; y < height - 1; y++)
                {
                    cNewMap[x, y] = new Complex (0, 0);

                    for (int i = -1; i < 2; i++)
                    {
                        for (int j = -1; j < 2; j++)
                        {
                            cNewMap[x, y] += cMap[x + i, y + j];
                        }
                    }

                    cNewMap[x, y] /= 9;
                }
            }

            return cNewMap;
        }

        /*public double Attenuate (int distc)
        {
            double sigma = 0.5;
            var commonDenom = 2.0d * sigma * sigma;
            var denominator = Math.Sqrt(2.0 * Math.PI) * sigma;

            var att = Math.Exp(distc * distc / commonDenom) / denominator;
            
            return att;
        }*/

        public double[,] Strengthen(Complex[,] cMap)
        {
            Complex[,] cNewMap = new Complex[width, height];
            Complex cNum = new Complex(0, 0);
            Complex cDenom = new Complex(0, 0);
            double[,] str = new double[width, height];
            double denom = 0;

            for (int x = 1; x < width - 1; x++)
            {
                for (int y = 1; y < height - 1; y++)
                {
                    cNum = new Complex(0, 0);
                    denom = 0;

                    for (int i = -1; i < 2; i++)
                    {
                        for (int j = -1; j < 2; j++)
                        {
                            cNum += cMap[x + i, y + j];

                            cDenom = cMap[x + i, y + j];

                            denom += Complex.Abs(cDenom);
                        }
                    }

                    str[x, y] = 1 - Complex.Abs(cNum) / denom;
                }
            }

            return str;
        }

        public double[,] Detect(double[,] vectMap)
        {
            Complex[,] cMap = new Complex[width, height];
            Complex[,] V_r = new Complex[width, height];
            double[,] str = new double[width, height];
            //double[, ,] V_e = new double[width, height, 2];

            //double gamma = 1;
            //double sum = 0;
            //int distc = 0;

            /*System.IO.StreamWriter file = new System.IO.StreamWriter(@"D:\Sin.txt");
            System.IO.StreamWriter file2 = new System.IO.StreamWriter(@"D:\Cos.txt");*/

            for (int x = 1; x < width - 1; ++x )
            {
                for (int y = 1; y < height - 1; ++y)
                {
                    cMap[x, y] = new Complex(Math.Sin(2*vectMap[x, y]), Math.Cos(2*vectMap[x, y]));

                    /*file.Write(Math.Round(newVectMap[x, y, 0], 2));
                    file.Write(" ");

                    file2.Write(Math.Round(newVectMap[x, y, 1], 2));
                    file2.Write(" ");*/
                }
                /*file.WriteLine();
                file2.WriteLine();*/
            }

            V_r = Regularize(cMap);
            str = Strengthen(V_r);

            /*for (int x = 1; x < width - 1; ++x)
            {
                for (int y = 1; y < height - 1; ++y)
                {
                    sum = 0;

                    for (int i = -1; i < 2; i++)
                    {
                        for (int j = -1; j < 1; j++)
                        {
                            sum += str[x + i, y + j];
                        }
                    }

                    distc = (int)(Math.Abs(width / 2 - x / 32) * Math.Abs(width / 2 - x / 32) + Math.Abs(height / 2 - y / 32) * Math.Abs(height / 2 - y / 32));

                    //V_e[x, y, 0] = V_r[x, y, 0] * (1 + gamma * Attenuate(distc) * sum / 9);
                    //V_e[x, y, 1] = V_r[x, y, 1] * (1 + gamma * Attenuate(distc) * sum / 9);
                }
            }*/

            return str;
        }

        public Bitmap MakeBitmap(double[,] byteMatrix)
        {
            Bitmap bmp = new Bitmap(width, height);

            for (int x = 0; x < width; ++x)
            {
                for (int y = 0; y < height; ++y)
                {
                    bmp.SetPixel(x, y, Color.FromArgb((int)byteMatrix[x, y], (int)byteMatrix[x, y], (int)byteMatrix[x, y]));
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
