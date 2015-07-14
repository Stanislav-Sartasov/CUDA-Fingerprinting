using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CUDAFingerprinting.Common.Tests
{
    [TestClass]
    public class NormalizeTests
    {
        [TestMethod]
        public void TestNormalization()
        {
            var bmp   = Resources.SampleFinger3;
            var array = ImageHelper.LoadImage(bmp);

            array = array.DoNormalization(100, 100);

            var mean = array.CalculateMean();
            var var  = array.CalculateVariation(mean);

            if (Math.Abs(mean - 100) / mean * 100 > 1) Assert.Fail();
            if (Math.Abs(var - 1000) / var * 100 > 1)  Assert.Fail();
        }
    }
}