using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CUDAFingerprinting.Common;

namespace CUDAFingerprinting.Common.SingularityRegionDetection
{
    public class SingularityRegionDetection
    {
        private int width;
        private int height;
        public SingularityRegionDetection(double[,] pic)
        {
            width = pic.GetLength(0);
            height = pic.GetLength(1);
        }

        public double Module(double[] v)
        {
            double sum = v[0] * v[0] + v[1] * v[1];
            return Math.Sqrt(sum);
        }

        public double[,,] Regularize(double[,,] vectMap)
        {
            double[, ,] newVectMap = new double[width, height, 2];
            double[] V_r = new double [2];

            for (int x = 1; x < width; x++)
            {
                for (int y = 1; y < height; y++)
                {
                    V_r[0] = 0;
                    V_r[1] = 0;
                    for (int i = -1; i < 2; i++)
                    {
                        for (int j = -1; j < 2; j++)
                        {
                            V_r[0] += vectMap[x + i, y + j, 0];
                            V_r[1] += vectMap[x + i, y + j, 1];
                        }
                    }

                    V_r[0] /= 9;
                    V_r[1] /= 9;

                    newVectMap[x, y, 0] = V_r[0];
                    newVectMap[x, y, 1] = V_r[1];
                }
            }

            return newVectMap;
        }

        public double Attenuate (int distc)
        {
            double sigma = 0.5;
            var commonDenom = 2.0d * sigma * sigma;
            var denominator = Math.Sqrt(2.0 * Math.PI) * sigma;

            var att = Math.Exp(distc * distc / commonDenom) / denominator;
            
            return att;
        }

        public double[,] Strengthen(double[,,] vectMap)
        {
            double[,,] newVectMap = new double[width, height, 2];
            double[] numVect = new double[2];
            double[] denomVect = new double[2];
            double[,] str = new double[width, height];
            double denom;

            for (int x = 1; x < width - 1; x++)
            {
                for (int y = 1; y < height - 1; y++)
                {
                    numVect[0] = 0;
                    numVect[1] = 0;
                    denom = 0;

                    for (int i = -1; i < 2; i++)
                    {
                        for (int j = -1; j < 1; j++)
                        {
                            numVect[0] += vectMap[x + i, y + j, 0];
                            numVect[1] += vectMap[x + i, y + j, 1];

                            denomVect[0] = vectMap[x + i, y + j, 0];
                            denomVect[1] = vectMap[x + i, y + j, 1];

                            denom += Module(denomVect);
                        }
                    }

                    str[x, y] = 1 - Module(numVect) / denom;
                }
            }

            return str;
        }

        public double[,,] Detect(double[,] vectMap)
        {
            double[, ,] newVectMap = new double[width, height, 2];
            double[, ,] V_r = new double[width, height, 2];
            double[,] str = new double[width, height];
            double[, ,] V_e = new double[width, height, 2];

            double gamma = 1;
            double sum = 0;
            int distc = 0;

            for (int x = 1; x < width - 1; ++x )
            {
                for (int y = 1; y < height - 1; ++y)
                {
                    newVectMap[x, y, 0] = Math.Sin(vectMap[x, y]);
                    newVectMap[x, y, 1] = Math.Cos(vectMap[x, y]);
                }
            }

            V_r = Regularize(newVectMap);
            str = Strengthen(V_r);

            for (int x = 1; x < width - 1; ++x)
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

                    V_e[x, y, 0] = V_r[x, y, 0] * (1 + gamma * Attenuate(distc) * sum / 9);
                    V_e[x, y, 1] = V_r[x, y, 1] * (1 + gamma * Attenuate(distc) * sum / 9);
                }
            }

            return V_e;
        }
    }
}
