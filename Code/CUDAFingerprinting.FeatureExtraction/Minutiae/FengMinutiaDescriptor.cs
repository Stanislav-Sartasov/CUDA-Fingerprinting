using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CUDAFingerprinting.Common;

namespace CUDAFingerprinting.FeatureExtraction.Minutiae
{
    public class FengMinutiaDescriptor
    {
        private static Descriptor Transformate(Descriptor desc_, Minutia center)
        {
            int i;
            Descriptor desc = new Descriptor();
            
            desc.Minutias = desc_.Minutias.ToList(); 
            desc.Center = desc_.Center;

            float angle = center.Angle - desc.Center.Angle;
            for (i = 0; i < desc.Minutias.Count; i++)
            {
                var min = desc.Minutias[i];

                min.X = Convert.ToInt32((desc.Minutias[i].X - desc.Center.X) * Math.Cos(angle) +
                            (+desc.Minutias[i].Y - desc.Center.Y) * Math.Sin(angle)) + center.X;
                min.Y = -(Convert.ToInt32((desc.Minutias[i].X - desc.Center.X) * Math.Sin(angle) +
                            (-desc.Minutias[i].Y + desc.Center.Y) * Math.Cos(angle)) - center.Y);
                min.Angle = MinutiaHelper.NormalizeAngle(min.Angle + angle);

                desc.Minutias[i] = min;
            }
            
            return desc;
        }

        private static Tuple<int, int> CountMatchings(Descriptor desc1, Descriptor desc2, int radius, int height, int width)
        {
            int m = 0, M = 0;
            int i, j;
            float eps = 0.1F;
            bool isExist;
            int r = 5;
            float fengConstant = 0.64F; //= 0.8 * 0.8;  0.8 is a magic constant!(from Feng book)

            for (i = 0; i < desc1.Minutias.Count; i++)
            {
                isExist = false;
                //sort desc2 and binary search is better solution
                for (j = 0; j < desc2.Minutias.Count; j++)
                {
                    if ((MinutiaHelper.SqrLength(desc1.Minutias[i], desc2.Minutias[j]) < r*r)
                        && (Math.Abs(desc1.Minutias[i].Angle - desc2.Minutias[j].Angle) < eps))
                    {
                        isExist = true;
                    }
                }

                if (isExist)
                {
                    ++m;
                    ++M;
                }
                else
                {
                    if ((MinutiaHelper.SqrLength(desc1.Minutias[i], desc2.Center) < fengConstant * radius * radius) &&
                        (desc1.Minutias[i].X >= 0 && desc1.Minutias[i].X < width
                        && desc1.Minutias[i].Y >= 0 && desc1.Minutias[i].Y < height))
                    {
                        ++M;
                    }
                }
            }

            return Tuple.Create(m, M);
        }

        public static float MinutiaCompare(Descriptor desc1, Descriptor desc2, int radius, int height, int width)
        {
            Descriptor tempdesc;
            Tuple<int, int> mM1, mM2;
            float s;
            tempdesc = Transformate(desc1, desc2.Center);
            mM1 = CountMatchings(tempdesc, desc2, radius, height, width);
            tempdesc = Transformate(desc2, desc1.Center);
            mM2 = CountMatchings(tempdesc, desc1, radius, height, width);
            s = (float)((mM1.Item1 + 1) * (mM2.Item1 + 1)) / (float)((mM1.Item2 + 1) * (mM2.Item2 + 1));
            return s;
        }

        public static float[,] DescriptorsCompare(List<Descriptor> descs1, List<Descriptor> descs2, int radius, int height, int width)
        {
            float[,] res = new float[descs1.Count, descs2.Count];
            int i, j;

            for (i = 0; i < descs1.Count; ++i)
            {
                for (j = 0; j < descs2.Count; ++j)
                {
                    res[i, j] = MinutiaCompare(descs1[i], descs2[j], radius, height, width);
                }
            }

            return res;
        }
    }
}