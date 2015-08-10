using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CUDAFingerprinting.Common;

namespace CUDAFingerprinting.FeatureExtraction.Minutiae
{
    public class DescriptorBuilder
    {
        private static float leng(Minutia p1, Minutia p2)
        {
            float length;
            length = (float)(Math.Pow(p1.X - p2.X, 2.0) +
                             Math.Pow(p1.Y - p2.Y, 2.0));
            return length;
        }

        public static List<Descriptor> BuildDescriptors(List<Minutia> list, int radius)
        {
            List<Descriptor> desc = new List<Descriptor>();
            int i, j;
            float length;
            for (i = 0; i < list.Count; i++)
            {
                Descriptor d = new Descriptor();
                for (j = i + 1; j < list.Count; j++)
                {
                    d.Center = list[i];
                    length = leng(list[i], list[j]);
                    if (length <= radius)
                    {
                        d.Minutias.Add(list[j]);
                    }
                }
                desc.Add(d);
            }
            return desc;
        }
    }
}
