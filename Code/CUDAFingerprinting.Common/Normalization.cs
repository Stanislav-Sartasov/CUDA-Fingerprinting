using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;

namespace CUDAFingerprinting.Common
{
    static class Normalization
    {
        public static double CalculateMean (this double [,] image)
        {
            int height = image.GetLength (1);
            int width  = image.GetLength (0);
            double mean = 0;
            for (int i = 0; i < height; i++) 
            {
                for (int j = 0; j < width; j++) 
                {
                    mean += image[i, j] / (height * width);
                }
            }
            return mean;
        }
        public static double CalculateVariation(this double[,] image, double mean)
        {
            int height = image.GetLength(1);
            int width  = image.GetLength(0);
            double variation = 0;
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    variation += Math.Pow ((image[i, j] - mean), 2) / (height * width);
                }
            }
            return variation;
        }
    }
}
