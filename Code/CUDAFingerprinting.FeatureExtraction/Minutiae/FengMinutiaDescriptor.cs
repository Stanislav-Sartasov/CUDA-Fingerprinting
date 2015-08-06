using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CUDAFingerprinting.Common;

namespace CUDAFingerprinting.FeatureExtraction.Minutiae
{
    class FengMinutiaDescriptor
    {
        private unsafe static Minutia[] Transformate(Minutia[] desc_, Minutia point1, Minutia point2)
        {
            int i;
            Minutia[] desc = new Minutia[desc_.Length];
            Array.Copy(desc_, desc, desc.Length);
            float angle = point2.Angle - point1.Angle;
            for (i = 0; i < desc.Length; i++)
            {
                desc[i].X = (desc[i].X - point1.X) * (int)Math.Cos(angle) +
                            (desc[i].Y - point1.Y) * (int)Math.Sin(angle) + point2.X;
                desc[i].Y = -(desc[i].X - point1.X) * (int)Math.Sin(angle) +
                            (desc[i].Y - point1.Y) * (int)Math.Cos(angle) + point2.Y;
                desc[i].Angle -= angle;
            }
            return desc;
        }

        private static Tuple<int, int> CountMatchings(Minutia[] desc1, Minutia[] desc2, float radius, int height, int width)
        {
            int m = 0, M = 0;
            int i, j;
            float eps = 0.1F;
            bool isExist;
            float magicConstant = 0.64F; //= 0.8 * 0.8  0.8 is magic constant too!

            for (i = 0; i < desc1.Length; i++)
            {
                isExist = false;

                for (j = 0; j < desc2.Length; j++)
                {
                    if ((desc1[i].X == desc2[j].X) && (desc1[i].Y == desc2[j].Y)
                        && (Math.Abs(desc1[i].Angle - desc2[j].Angle) <= eps))
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
                    if ((Math.Abs(desc1[i].X - desc2[j].X) +
                        Math.Abs(desc1[i].Y - desc2[j].Y) < magicConstant * radius * radius) &&
                        (desc1[i].X >= 0 && desc1[i].Y < width && desc1[i].Y >= 0 && desc1[i].Y < height))
                    {
                        ++M;
                    }
                }
            }

            return Tuple.Create(m, M);
        }

        public static float MinutiaCompare(Minutia[] desc1, Minutia point1, Minutia[] desc2, Minutia point2, float radius, int height, int width)
        {
            Minutia[] tempdesc = Transformate(desc1, point1, point2);


            return 0.0F;
        }
    }
}