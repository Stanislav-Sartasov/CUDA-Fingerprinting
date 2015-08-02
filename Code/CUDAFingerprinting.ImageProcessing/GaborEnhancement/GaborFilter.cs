using System;

namespace CUDAFingerprinting.ImageProcessing.GaborEnhancement
{
    internal class GaborFilter
    {
        public ImageProcessing.GaborEnhancement.Filter[,] Filters;
        //public int Count;
        public const int FrequencyCount = 4;
        public static double[] FrequencyMatrix = 
        {
            1.0/25.0,
            1.0/16.0,
            1.0/9.0,
            1.0/3.0
        };
      //  public static double[] FrequencyMatrix = new double[FrequencyCount];
        public GaborFilter(int angleCount, int size)
        {
            var bAngle = Math.PI / angleCount;
            Filters = new ImageProcessing.GaborEnhancement.Filter[angleCount, FrequencyCount];

            //for (int i = 0; i < FrequencyCount; i++)
            //{
            //    FrequencyMatrix[i] = 1.0 / 9.0;
            //}
            for (int i = 0; i < angleCount; i++)
            {
                for (int j = 0; j < FrequencyCount; j++)
                {
                    Filters[i, j] = new ImageProcessing.GaborEnhancement.Filter(size, bAngle*i, FrequencyMatrix[j]);
                }
            }
        }
    }
}
