using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;

namespace CUDAFingerprinting.Common
{
    class ImageEnhancement
    {
        public static double[,] Enhance(double[,] img, double[,] orientMatrix, double frequency, int filterSize,
            float[] angles)
        {
            double[,] result = new double[img.GetLength(0),img.GetLength(1)];
            for (int i = 0; i < img.GetLength(0); i++)
                for (int j = 0; j < img.GetLength(1); j++)
                {
                    double enhancedPxl = 0;
                    for (int u = -filterSize/2; u <= filterSize/2; u++)
                        for (int v = -filterSize/2; v <= filterSize/2; v++)
                        {
                            double diff = Double.MaxValue;
                            double angle = 0;
                            for (int angInd = 0; angInd < angles.Length; angInd++)
                                if (Math.Abs(angles[angInd] - img[i, j]) < diff)
                                {
                                    angle = angles[angInd];
                                    diff = Math.Abs(angles[angInd] - img[i, j]);
                                }
                            enhancedPxl += Gabor(u, v, angle, frequency) * img[i - u, j - v];//Insert real Gabor filtering function name.
                        }
                    result[i, j] = enhancedPxl;
                }
            return result;
        }
    }
}
