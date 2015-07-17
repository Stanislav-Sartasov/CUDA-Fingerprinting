using System;
using System.Drawing;
using System.Globalization;
using System.IO;
using CUDAFingerprinting.Common;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CUDAFingerprinting.FeatureExtraction.Tests
{
    [TestClass]
    public class AdaptiveBinarizationTest
    {
        [TestMethod]
        public void AdaptiveBinarizationTestMethod()
        {
            int[,] arrayI = ImageHelper.LoadImageAsInt(Resources._1test);
            var arr = AdaptiveBinarization.AdaptiveBinarize(arrayI);
            ImageHelper.SaveArrayToBitmap(arr).Save(Path.GetTempPath() + Guid.NewGuid() + ".bmp");
        }
    }
}
