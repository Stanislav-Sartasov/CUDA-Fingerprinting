using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Drawing;
using CUDAFingerprinting.Common;
using System.Diagnostics;
using CUDAFingerprinting.FeatureExtraction.Minutiae;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CUDAFingerprinting.FeatureExtraction.Tests
{
    [TestClass]
    public class FengMinutiaDescriptorTest
    {
        private static List<Minutia> readMinutiae(string s)
        {
            float[] mas = s.Split(new char[] { ' ', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries).Select(x => float.Parse(x)).ToArray();
            
            int n = (int)mas[0];

            List<Minutia> mins = new List<Minutia>();
            Minutia m = new Minutia();
            
            for (int i = 1; i <= 3*n; i +=3)
            {
                m.X = (int)mas[i];
                m.Y = (int)mas[i + 1];
                m.Angle = MinutiaHelper.NormalizeAngle(mas[i + 2]);

                mins.Add(m);
            }

            return mins;
        }

        public static void getImgSize(Bitmap map, ref int heigth, ref int width)
        {
            int[,] img = ImageHelper.LoadImage<int>(map);
            heigth = img.GetLength(1);
            width = img.GetLength(0);
        }
        [TestMethod, Description("Compare two handwriting descriptors")]
        public void DescriptorsCompareTestOneToOne()
        {
            int i;
            float s;
            float eps = 0.05F;
            float check = 1.0F;
            int radius = 10;
            int height = 20;
            int width = 20;
            const int line = 5;
            List<Minutia> ms1 = new List<Minutia>();
            List<Minutia> ms2 = new List<Minutia>();
            Minutia c1, c2;
            c1.X = 0;
            c1.Y = 0;
            c1.Angle = 0.0F;
            c2.X = 6;
            c2.Y = 6;
            c2.Angle = (float)Math.PI;
            for (i = 0; i < line; i++)
            {
                Minutia m1 = new Minutia();
                Minutia m2 = new Minutia();
                m1.X =1 + i + c1.X;
                m1.Y =1 + i + c1.Y;
                m1.Angle = 0.5F * (float)Math.PI;
                m2.X =-1 - i + c2.X;
                m2.Y =-1 - i + c2.Y;
                m2.Angle = 1.5F * (float)Math.PI;
                ms1.Add(m1);
                ms2.Add(m2);
                
            }
            Descriptor desc1 = new Descriptor(ms1, c1);
            Descriptor desc2 = new Descriptor(ms2, c2);

            s = FengMinutiaDescriptor.MinutiaCompare(desc1, desc2, radius, height, width);
            Assert.IsTrue(Math.Abs(s - check) < eps);
        }

        private bool cmp(List<Minutia> mins1, List<Minutia> mins2, bool expected)
        {
            int heigth = 0;
            int width = 0;
            getImgSize(Resources.minutia1_11, ref heigth, ref width);

            int radius = 70;
            float eps = 0.01F;

            List<Descriptor> desc1 = DescriptorBuilder.BuildDescriptors(mins1, radius);
            List<Descriptor> desc2 = DescriptorBuilder.BuildDescriptors(mins2, radius);
            var s = FengMinutiaDescriptor.DescriptorsCompare(desc1, desc2, radius, heigth, width);

            var res = MinutiaeMatching.MatchMinutiae(s, mins1, mins2);

            bool flag;

            float border = 0.3F; //why? JUST FOR FUN
            float cmp = (float)(2.0 * res.Count) / (mins1.Count + mins2.Count);
            flag = cmp > border;

            return flag == expected;
        }

        [TestMethod, Description("Compare set of descriptors with itself")]
        public void DescriptorsCompareTestManyToManyOneFingerprint()
        {
            bool cmpres;
            List<Minutia> mins1 = readMinutiae(Resources.minutia1_1);
            List<Minutia> mins2 = readMinutiae(Resources.minutia1_1);
            cmpres = cmp(mins1, mins2, true);
            Assert.IsTrue(cmpres);
        }
        [TestMethod, Description("Compare sets of descriptors from different fingers")]
        public void DescriptorsCompareTestManyToManyDifferentFingers1()
        {
            bool cmpres;
            List<Minutia> mins1 = readMinutiae(Resources.minutia1_1);
            List<Minutia> mins2 = readMinutiae(Resources.minutia2_2);
            cmpres = cmp(mins1, mins2, false);
            Assert.IsTrue(cmpres);
        }

        [TestMethod, Description("Compare sets of descriptors from different fingers")]
        public void DescriptorsCompareTestManyToManyDifferentFingers2()
        {
            bool cmpres;
            List<Minutia> mins1 = readMinutiae(Resources.minutia1_1);
            List<Minutia> mins2 = readMinutiae(Resources.minutia2_6);
            cmpres = cmp(mins1, mins2, false);
            Assert.IsTrue(cmpres);
        }

        [TestMethod, Description("Compare sets of descriptors from different fingerprints of one finger")]
        public void DescriptorsCompareTestManyToManyDifferentFingerprints1()
        {
            bool cmpres;
            List<Minutia> mins1 = readMinutiae(Resources.minutia2_6);
            List<Minutia> mins2 = readMinutiae(Resources.minutia2_2);
            cmpres = cmp(mins1, mins2, true);
            Assert.IsTrue(cmpres);
        }

        [TestMethod, Description("Compare sets of descriptors from different fingerprints of one finger")]
        public void DescriptorsCompareTestManyToManyDifferentFingerprints2()
        {
            bool cmpres;
            List<Minutia> mins1 = readMinutiae(Resources.minutia2_6);
            List<Minutia> mins2 = readMinutiae(Resources.minutia2_3);
            cmpres = cmp(mins1, mins2, true);
            Assert.IsTrue(cmpres);
        }
    }
}
