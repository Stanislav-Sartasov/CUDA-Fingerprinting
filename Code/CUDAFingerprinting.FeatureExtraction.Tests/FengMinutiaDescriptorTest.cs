using System;
using System.IO;
using System.Collections.Generic;
using CUDAFingerprinting.Common;
using CUDAFingerprinting.FeatureExtraction.Minutiae;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CUDAFingerprinting.FeatureExtraction.Tests
{
    [TestClass]
    public class FengMinutiaDescriptorTest
    {
        [TestMethod]
        public void FengMinutiaDescriptorTestMethod()
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

        [TestMethod]
        public void DescriptorsCompareTest1()
        {
            System.IO.StreamReader file = new System.IO.StreamReader("D:\\test2.txt");
        }
    }
}
