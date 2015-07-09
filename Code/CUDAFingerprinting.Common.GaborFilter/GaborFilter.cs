using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CUDAFingerprinting.Common.GaborFilter
{
    class GaborFilter
    {
        public Filter[] Filters;
        public int Count;

        public GaborFilter(int count, int size)
        {
            var bAngle = Math.PI / count;
            Filters = new Filter[count];

            for (int i = 0; i < count; i++)
            {
                Filters[i] = new Filter(size, bAngle * i);
            }
        }
    }
}
