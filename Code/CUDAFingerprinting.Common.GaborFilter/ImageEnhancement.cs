using System;
using CUDAFingerprinting.Common.GaborFilter;
namespace CUDAFingerprinting.Common.GaborFilter
{
    class ImageEnhancement
    {
        public static double[,] Enhance(double[,] img, double[,] orientMatrix, double frequency, int filterSize,
            int angleNum)
        {
            double[,] result = new double[img.GetLength(0),img.GetLength(1)];
            double [] angles = new double[angleNum];
            double constAngle = Math.PI/angleNum;
            for (int i = 0; i < angleNum; i++)
                angles[i] = constAngle*i;
            var gabor = new GaborFilter(angleNum, filterSize);
            for (int i = 0; i < img.GetLength(0); i++)
                for (int j = 0; j < img.GetLength(1); j++)
                {
                    double enhancedPxl = 0;
                    for (int u = -filterSize/2; u <= filterSize/2; u++)
                        for (int v = -filterSize/2; v <= filterSize/2; v++)
                        {
                            double diff = Double.MaxValue;
                            int angle = 0;
                            for (int angInd = 0; angInd < angleNum; angInd++)
                                if (Math.Abs(angles[angInd] - img[i, j]) < diff)
                                {
                                    angle = angInd;
                                    diff = Math.Abs(angles[angInd] - img[i, j]);
                                }
                            enhancedPxl += gabor.Filters[angle].Matrix[u, v] * img[i - u, j - v];
                        }
                    result[i, j] = enhancedPxl;
                }
            return result;
        }
    }
}
