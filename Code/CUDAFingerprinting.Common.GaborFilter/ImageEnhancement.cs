using System;
using System.Globalization;

namespace CUDAFingerprinting.Common.GaborFilter
{
    public class ImageEnhancement
    {
        public static double[,] Enhance(double[,] img, double[,] orientMatrix, double frequency, int filterSize,
            int angleNum)
        {
            int imgHeight = img.GetLength(0);
            int imgWidth  = img.GetLength(1);
            double[,] result = new double[imgHeight,imgWidth];

            double [] angles = new double[angleNum];
            double constAngle = Math.PI/angleNum;
            for (int i = 0; i < angleNum; i++)
                angles[i] = constAngle * i - Math.PI / 2;

            var gabor = new GaborFilter(angleNum, filterSize, frequency);
            int center = filterSize / 2; //filter is always a square.
            int upperCenter = (filterSize & 1) == 0 ? center - 1 : center;

            for (int i = 0; i < imgHeight; i++)
            {
                for (int j = 0; j < imgWidth; j++)
                {
                    double diff = Double.MaxValue;
                    int angle = 0;
                    for (int angInd = 0; angInd < angleNum; angInd++)
                        if (Math.Abs(angles[angInd] - orientMatrix[i, j]) < diff)
                        {
                            angle = angInd;
                            diff = Math.Abs(angles[angInd] - orientMatrix[i, j]);
                        }
                    for (int u = -upperCenter; u <= center; u++)
                    {
                        for (int v = -upperCenter; v <= center; v++)
                        {
                          

                            int indexX = i + u;
                            int indexY = j + v;
                            if (indexX < 0) indexX = 0;
                            if (indexX >= imgHeight) indexX = imgHeight - 1;
                            if (indexY < 0) indexY = 0;
                            if (indexY >= imgWidth) indexY = imgWidth - 1;
                            result[i, j] += gabor.Filters[angle].Matrix[center - u, center - v]*img[indexX, indexY];
                        }
                    }
                    result[i, j] = result[i, j] > 255 ? 255 : (result[i, j] < 0 ? 0 : result[i, j]);
                }
            }
            return result;
        }
    }
}
