using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CUDAFingerprinting.Common;

namespace CUDAFingerprinting.FeatureExtraction.Minutiae
{
    class DescriptorBuilder
    {
        private float leng(Minutia p1, Minutia p2)
        {
            float length;
            length = (float)(Math.Pow(p1.X - p2.X, 2.0) +
                             Math.Pow(p1.Y - p2.Y, 2.0));
            return length;
        }

        public List<Descriptor> BuildDescriptors(List<Minutia> list, int radius)
        {
            List<Descriptor> desc = new List<Descriptor>();
            int i, j;
            float length;
            for (i = 0; i < list.Count; i++)
            {
                for (j = i + 1; j < list.Count; j++)
                {
                    length = leng(list[i], list[j]);
                    if (length <= radius)
                    {
                        desc[i].Minutias.Add(list[j]);
                    }
                }
            }
            return desc;
        }
    }
}
