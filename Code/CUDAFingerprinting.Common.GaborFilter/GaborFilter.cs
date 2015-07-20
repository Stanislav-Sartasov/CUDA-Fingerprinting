using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CUDAFingerprinting.Common.GaborFilter
{
    internal class GaborFilter
    {
        public Filter[,] Filters;
        //public int Count;
        private const int FrequencyCount = 4;
        private double[] FrequencyMatrix = 
        {
            1.0/25.0,
            1.0/16.0,
            1.0/9.0,
            1.0/3.0
        };
        public GaborFilter(int angleCount, int size)
        {
            var bAngle = Math.PI / angleCount;
            Filters = new Filter[angleCount, FrequencyCount];

            for (int i = 0; i < angleCount; i++)
            {
                for (int j = 0; j < FrequencyCount; j++)
                {
                    Filters[i, j] = new Filter(size, bAngle*i, FrequencyMatrix[j]);
                }
            }
        }
    }
}
