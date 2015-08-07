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
            Minutia[] mins = new Minutia[desc_.Minutias.Length];
            Array.Copy(desc_.Minutias, mins, desc_.Minutias.Length);
            desc.Minutias = mins;
            desc.Center = desc_.Center;

            float angle = center.Angle - desc.Center.Angle;
            for (i = 0; i < desc.Minutias.Length; i++)
            {
                desc.Minutias[i].X = (int)Math.Round((desc.Minutias[i].X - desc.Center.X) * Math.Cos(angle) -
                            (desc.Minutias[i].Y - desc.Center.Y) * Math.Sin(angle)) + center.X;
                desc.Minutias[i].Y = (int)Math.Round((desc.Minutias[i].X - desc.Center.X) * Math.Sin(angle) +
                            (desc.Minutias[i].Y - desc.Center.Y) * Math.Cos(angle)) + center.Y;
                desc.Minutias[i].Angle += angle;
            }
            
            return desc;
        }

        private static Tuple<int, int> CountMatchings(Descriptor desc1, Descriptor desc2, int radius, int height, int width)
        {
            int m = 0, M = 0;
            int i, j;
            float eps = 0.1F;
            bool isExist;
            float fengConstant = 0.64F; //= 0.8 * 0.8;  0.8 is a magic constant!(from Feng book)

            for (i = 0; i < desc1.Minutias.Length; i++)
            {
                isExist = false;
                //sort desc2 and binary search is better solution
                for (j = 0; j < desc2.Minutias.Length; j++)
                {
                    if ((desc1.Minutias[i].X == desc2.Minutias[j].X) && (desc1.Minutias[i].Y == desc2.Minutias[j].Y)
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
                    if ((Math.Pow(desc1.Minutias[i].X - desc2.Center.X, 2) +
                        Math.Pow(desc1.Minutias[i].Y - desc2.Center.Y, 2) < fengConstant * radius * radius) ||
                        (desc1.Minutias[i].X >= 0 && desc1.Minutias[i].Y < width
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

        public static float[,] DescriptorsCompare(Descriptor[] descs1, Descriptor[] descs2, int radius, int height, int width)
        {
            float[,] res = new float[descs1.Length, descs2.Length];
            int i, j;

            for (i = 0; i < descs1.Length; ++i)
            {
                for (j = 0; j < descs2.Length; ++j)
                {
                    res[i, j] = MinutiaCompare(descs1[i], descs2[j], radius, height, width);
                }
            }

            return res;
        }
    }
}