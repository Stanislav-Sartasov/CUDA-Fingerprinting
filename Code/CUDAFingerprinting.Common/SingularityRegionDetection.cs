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

        public Complex[,] Regularize(Complex[,] cMap)
        {
            Complex[,] cNewMap = new Complex[width, height];

            for (int x = 3; x < width - 3; x++)
            {
                for (int y = 3; y < height - 3; y++)
                {
                    cNewMap[x, y] = new Complex (0, 0);

                    for (int i = -1; i < 2; i++)
                    {
                        for (int j = -1; j < 2; j++)
                        {
                            cNewMap[x, y] += cMap[x + i, y + j];
                        }
                    }

                    cNewMap[x, y] /= 441;
                }
            }

            return cNewMap;
        }

        public double[,] Strengthen(Complex[,] cMap)
        {
            Complex[,] cNewMap = new Complex[width, height];
            Complex cNum = new Complex(0, 0);
            Complex cDenom = new Complex(0, 0);
            double[,] str = new double[width, height];
            double denom = 0;

            for (int x = 24; x < width - 24; x++)
            {
                for (int y = 24;  y < height - 24; y++)
                {
                    cNum = new Complex(0, 0);
                    denom = 0;

                    for (int i = -24; i < 23; i++)
                    {
                        for (int j = -24; j < 23; j++)
                        {
                            cNum += cMap[x + i, y + j];

                            cDenom = cMap[x + i, y + j];

                            denom += cMap[x + i, y + j].Magnitude;
                        }
                    }


                    str[x, y] = 1 - cNum.Magnitude / denom;
                      
                }
            }

            return str;
        }

        public double[,] Detect(double[,] vectMap)
        {
            Complex[,] cMap = new Complex[width, height];
            Complex[,] V_r = new Complex[width, height];
            double[,] str = new double[width, height];

            for (int x = 1; x < width - 1; ++x )
            {
                for (int y = 1; y < height - 1; ++y)
                {
                    cMap[x, y] = new Complex(Math.Sin(2*vectMap[x, y]), Math.Cos(2*vectMap[x, y]));
                }
            }

            V_r = Regularize(cMap);

            var newField = V_r.Select2D(x => x.Phase / 2);

            str = Strengthen(V_r);

            return str;
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
