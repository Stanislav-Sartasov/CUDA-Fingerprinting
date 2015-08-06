using System;
using System.IO;
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
            float radius = 10.0F;
            int height = 20;
            int width = 20;
            const int line = 5;
            Minutia[] ms1 = new Minutia[line];
            Minutia[] ms2 = new Minutia[line];
            Descriptor desc1, desc2;
            desc1.Center.X = 0;
            desc1.Center.Y = 0;
            desc1.Center.Angle = 0.0F;
            desc2.Center.X = 6;
            desc2.Center.Y = 6;
            desc2.Center.Angle = (float)Math.PI;
            for (i = 0; i < line; i++)
            {
                ms1[i].X =1 + i + desc1.Center.X;
                ms1[i].Y =1 + i + desc1.Center.Y;
                ms2[i].X =-1 - i + desc2.Center.X;
                ms2[i].Y =-1 - i + desc2.Center.Y;
            }
            desc1.Minutias = ms1;
            desc2.Minutias = ms2;
            s = FengMinutiaDescriptor.MinutiaCompare(desc1, desc2, radius, height, width);
            Assert.AreEqual(s, 1.0F);
        }
    }
}
