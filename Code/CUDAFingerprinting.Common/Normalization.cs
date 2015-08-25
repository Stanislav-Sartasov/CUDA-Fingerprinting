using System;

namespace CUDAFingerprinting.Common
{
    static public class Normalization
    {
        static public double CalculateMean(this double[,] image)
        {
            int height = image.GetLength(1);
            int width = image.GetLength(0);
            double mean = 0;
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    mean += image[i, j] / (height * width);
                }
            }
            return mean;
        }

        static public double CalculateVariation(this double[,] image, double mean)
        {
            int height = image.GetLength(1);
            int width = image.GetLength(0);
            double variation = 0;
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    variation += Math.Pow((image[i, j] - mean), 2) / (height * width);
                }
            }
            return variation;
        }

        static public double[,] DoNormalization(this double[,] image, int bordMean, int bordVar)
        {
            var mean = image.CalculateMean();
            var variation = image.CalculateVariation(mean);

            for (int i = 0; i < image.GetLength(0); i++)
            {
                for (int j = 0; j < image.GetLength(1); j++)
                {
                    if (image[i, j] > mean)
                    {
                        image[i, j] = bordMean + Math.Sqrt((bordVar * Math.Pow(image[i, j] - mean, 2)) / variation);
                    }
                    else
                    {
                        image[i, j] = bordMean - Math.Sqrt((bordVar * Math.Pow(image[i, j] - mean, 2)) / variation);
                    }
                }
            }

            return image;
        }
    }
}
